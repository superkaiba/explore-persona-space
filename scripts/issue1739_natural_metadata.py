"""Build compact, source-verified natural-extraction metadata from local caches.

This does not generate, judge or score any answer. It packages archived labels,
unaltered prompt content, manual eligibility and the frozen Figure 6 reference.
"""

from __future__ import annotations

# ruff: noqa: E402 -- source-root bootstrap precedes project imports
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.issue1739_natural_score import sha, write_json

SHARED = Path("/home/thomasjiralerspong/explore-persona-space")
RAW = Path("/dev/shm/issue1739-fixed-transfer/inputs/raw_metadata")
PROVENANCE = Path("/dev/shm/issue1739-fixed-transfer/inputs/provenance")


def normalized_hash(text):
    """Use the verified historical #1739 lowercase/whitespace hash recipe."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty prompt text")
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def jsonl(path):
    """Read JSONL by physical lines; Unicode line separators are valid content."""
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def pack_metadata(metadata, target):
    """Keep every text artifact below the repository's 9 MB upload ceiling."""
    contexts = metadata.pop("contexts")
    parts, current, size = [], {}, 2

    def flush():
        nonlocal current, size
        if not current:
            return
        name = f"contexts_{len(parts):03d}.json"
        data = json.dumps(current, separators=(",", ":")).encode()
        path = target / name
        path.write_bytes(data)
        parts.append({"path": name, "sha256": sha(path), "n": len(current)})
        current, size = {}, 2

    for cid, row in contexts.items():
        encoded = json.dumps({cid: row}, separators=(",", ":")).encode()
        if len(encoded) > 8_000_000:
            raise ValueError(f"Single context exceeds upload shard ceiling: {cid}")
        if size + len(encoded) > 8_000_000:
            flush()
        current[cid] = row
        size += len(encoded)
    flush()
    metadata["context_parts"] = parts
    metadata["n_contexts"] = len(contexts)
    write_json(target / "metadata.json", metadata)
    if (target / "metadata.json").stat().st_size >= 9_000_000:
        raise ValueError("Metadata index exceeds upload ceiling")


def build(dest, eligibility_path, reference_path):  # noqa: C901 -- source-format validation
    """Validate the source labels and publish one self-contained bundle per trait."""
    dest = Path(dest)
    eligibility = json.loads(Path(eligibility_path).read_text())
    exclusions = eligibility["excluded_contexts"]
    if sha(reference_path) != "ba6dda3bd3ca206e65725085532e40b277e0cbd1b4a2f1d0b2a72c0158cb2384":
        raise ValueError("Current Figure6 summary differs from frozen source")
    references = json.loads(Path(reference_path).read_text())
    contexts = {}
    source_files = {
        str(eligibility_path): sha(eligibility_path),
        str(reference_path): sha(reference_path),
    }
    prompt_index_path = PROVENANCE / "prompt_index.jsonl"
    index = {}
    for row in jsonl(prompt_index_path):
        if row["namespace"].endswith("_labeling") or row["namespace"] == "wildchat":
            index[row["context_id"]] = row
    source_files[str(prompt_index_path)] = sha(prompt_index_path)
    paths = sorted(RAW.glob("original/labeling_*.shard*.jsonl"))
    paths += sorted(RAW.glob("wildchat/*.jsonl"))
    paths += sorted(
        (
            SHARED / "data/issue_1739/syco_ood_vm/pack_main/issue1739_ctxmap/"
            "syco_ood/raw_completions/main"
        ).glob("root.shard*.jsonl")
    )
    paths += sorted(
        (SHARED / "data/issue_1739/evil_ood_full_stage").glob("*/rollouts/full/*_seed0.json")
    )
    if not paths:
        raise ValueError("No raw generation metadata")
    for path in paths:
        source_files[str(path)] = sha(path)
        records = [json.loads(path.read_text())] if path.suffix == ".json" else jsonl(path)
        for packed in records:
            doc = packed.get("doc", packed)
            if "context_id" not in doc:
                continue  # explicit aggregate manifest, never a generation
            cid = doc["context_id"]
            query, prompt = doc["query"], doc["prompt_text"]
            qhash = normalized_hash(query)
            phash = normalized_hash(prompt)
            if cid in contexts:
                if contexts[cid]["normalized_sha256"] != phash:
                    raise ValueError(f"Prompt changed across cached responses: {cid}")
                contexts[cid]["rollout_ks"].append(doc["rollout_k"])
                continue
            if cid in index and (
                qhash != index[cid]["question_normalized_sha256"]
                or phash != index[cid]["normalized_sha256"]
            ):
                raise ValueError(f"Provenance hash mismatch: {cid}")
            import re

            users = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S)
            if not users:
                raise ValueError(f"Missing rendered user turn: {cid}")
            natural_source = (
                doc["behavior"] in {"sycophancy", "hallucination"} and doc["split"] == "train"
            ) or (doc["behavior"] == "evil" and doc["rung"] in {"hhrt", "toxicchat"})
            contexts[cid] = {
                "context_id": cid,
                "query": query,
                "prompt_text": prompt,
                "question_normalized_sha256": qhash,
                "normalized_sha256": phash,
                "user_turn_hashes": [normalized_hash(u) for u in users],
                "natural_eligible": natural_source and cid not in exclusions,
                "natural_source": natural_source,
                "exclusion_reason": exclusions.get(cid),
                "rollout_ks": [doc["rollout_k"]],
                "generation_meta": doc["meta"],
                "source_path": str(path),
                "source_record": packed.get("src"),
                "rung": doc["rung"],
                "split": doc["split"],
                "behavior": doc["behavior"],
            }
    for cid, audit in {**eligibility["reviewed_retained_contexts"], **exclusions}.items():
        if cid not in contexts:
            raise ValueError(f"Audited prompt absent: {cid}")
        exact = hashlib.sha256(contexts[cid]["query"].encode()).hexdigest()
        if exact != audit["query_sha256"]:
            raise ValueError(f"Eligibility query changed: {cid}")
    for name, corpus in eligibility["corpora"].items():
        rows = {
            cid: row
            for cid, row in contexts.items()
            if row["natural_source"]
            and (
                (name == "evil_eval_natural" and row["behavior"] == "evil")
                or name == f"{row['behavior']}_train"
            )
        }
        fingerprint = hashlib.sha256(
            "".join(
                f"{cid}\t{hashlib.sha256(rows[cid]['query'].encode()).hexdigest()}\n"
                for cid in sorted(rows)
            ).encode()
        ).hexdigest()
        if (
            len(rows) != corpus["n_contexts"]
            or fingerprint != corpus["context_query_sha256_fingerprint"]
            or sha(corpus["label_path"]) != corpus["label_sha256"]
        ):
            raise ValueError(f"Eligibility corpus fingerprint mismatch: {name}")
    hallu = json.loads((PROVENANCE / "hallucination_per_rollout.json").read_text())
    source_files[str(PROVENANCE / "hallucination_per_rollout.json")] = sha(
        PROVENANCE / "hallucination_per_rollout.json"
    )
    bundles = {}
    for behavior in ("evil", "sycophancy", "hallucination"):
        target = dest / behavior
        target.mkdir(parents=True, exist_ok=True)
        history_key = (
            f"5aae0a472b:eval_results/issue_1739/r2v2_fits/{behavior}/all_arms_spearman.json"
        )
        history_bytes = subprocess.check_output(["git", "show", history_key], cwd=ROOT)
        if hashlib.sha256(history_bytes).hexdigest() != references["inputs"][history_key]:
            raise ValueError(f"Frozen Figure6 source hash mismatch: {history_key}")
        history = json.loads(history_bytes)
        expected_hashes = set(history["meta"]["input_sha256"].values())
        label_paths = {
            "train": SHARED / f"eval_results/issue_1739/dv_dataset/{behavior}/labeling.json",
            "wildchat": SHARED
            / f"eval_results/issue_1739/wildchat_rung/dv_dataset/{behavior}/labeling.json",
        }
        if behavior != "hallucination":
            subdir = "evil_ood_full" if behavior == "evil" else "syco_ood"
            label_paths["ood"] = (
                SHARED / f"eval_results/issue_1739/{subdir}/dv_dataset/{behavior}/labeling.json"
            )
        all_labels = []
        for name, path in label_paths.items():
            digest = sha(path)
            if digest not in expected_hashes:
                raise ValueError(f"Label source differs from Figure 6: {path} {digest}")
            source_files[str(path)] = digest
            shutil.copyfile(path, target / f"{name}_labels.json")
            all_labels.extend(json.loads(path.read_text())["rows"])
        keep = {r["context_id"] for r in all_labels if r.get("dv") is not None}
        missing = sorted(keep - contexts.keys())
        if missing:
            raise ValueError(f"Missing {len(missing)} prompt records: {behavior}: {missing[:5]}")
        score_source = hallu["rows"] if behavior == "hallucination" else all_labels
        scores = {
            r["context_id"]: r["per_rollout_scores"]
            for r in score_source
            if "per_rollout_scores" in r
        }
        if behavior == "hallucination":
            lookup = {r["context_id"]: r for r in hallu["rows"]}
            for row in all_labels:
                if row["context_id"] in lookup and row.get("dv") != lookup[row["context_id"]].get(
                    "dv"
                ):
                    raise ValueError(f"Hallucination per-response DV mismatch: {row['context_id']}")
        metadata = {
            "behavior": behavior,
            "contexts": {c: contexts[c] for c in sorted(keep)},
            "scores": scores,
            "source_files": source_files,
            "eligibility_sha256": sha(eligibility_path),
            "historical_answer_pooling": "all cached responses (original Figure6)",
            "extraction_answer_pooling": "valid judged responses, equal prompt weights",
        }
        pack_metadata(metadata, target)
        write_json(
            target / "reference.json",
            {**references, "cells": [c for c in references["cells"] if c["behavior"] == behavior]},
        )
        bundles[behavior] = {
            p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in target.iterdir()
            if p.is_file()
        }
    write_json(dest / "bundle_manifest.json", {"files": bundles, "source_files": source_files})
    return bundles


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: issue1739_natural_metadata.py OUTPUT ELIGIBILITY_JSON REFERENCE_JSON"
        )
    print(json.dumps(build(*sys.argv[1:]), indent=2))
