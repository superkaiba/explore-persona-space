"""Stage immutable prompt/rollout metadata for the matched fixed-direction run.

The packed source shards are retained, verified against both the pinned Hub
objects and their packing manifests, and reduced to a text-free context index.
Hash normalization matches the #779 parent corpus: lowercase and collapse
whitespace, without Unicode normalization. Full rendered prompts and bare user
questions are hashed separately; the #779 manifest contains bare questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.issue1739_covariance_stage import (  # noqa: E402
    REPO,
    REVISION,
    Progress,
    scoped_entries,
    sha256,
    verified_file,
    write_json,
)

BEHAVIORS = ("evil", "sycophancy", "hallucination")
PREFIXES = {
    "original": "issue1739_ctxmap/raw_completions",
    "wildchat": "issue1739_ctxmap/wildchat_rung/raw_completions_packed",
}
JUDGE_PATH = "issue1739_ctxmap/judge/hallucination/labeling_per_rollout.json"
HASH_RECIPE = "UTF-8 SHA256; normalized text = ' '.join(text.lower().split()); no NFC"


def text_hashes(text: str) -> tuple[str, str]:
    """Return exact and parent-corpus-compatible normalized text hashes."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt/question must be a nonempty string")
    normalized = " ".join(text.lower().split())
    return (
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


def user_turn_hashes(prompt: str) -> list[dict]:
    """Hash stripped Qwen user turns, matching #779's first-user extraction.

    This is a rendered-template audit, not a tokenizer reconstruction. Literal
    template markers inside content can produce conservative extra matches.
    """
    turns = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, flags=re.DOTALL)
    if not turns:
        raise ValueError("rendered prompt has no complete Qwen user turn")
    hashes = []
    for index, turn in enumerate(turns):
        exact, normalized = text_hashes(turn.strip())
        hashes.append({"turn_index": index, "exact_sha256": exact, "normalized_sha256": normalized})
    return hashes


def verify_hub_file(path: Path, entry) -> dict:
    """Verify bytes against the immutable LFS SHA or ordinary Git blob hash."""
    if path.stat().st_size != entry.size:
        raise ValueError(f"Hub source size mismatch: {path}")
    file_sha = sha256(path)
    if entry.lfs is not None:
        if file_sha != entry.lfs.sha256:
            raise ValueError(f"Hub LFS SHA256 mismatch: {path}")
    else:
        digest = hashlib.sha1(f"blob {entry.size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != entry.blob_id:
            raise ValueError(f"Hub Git blob mismatch: {path}")
    return {
        "path_in_repo": entry.path,
        "bytes": entry.size,
        "sha256": file_sha,
        "git_blob_id": entry.blob_id,
        "lfs_sha256": entry.lfs.sha256 if entry.lfs is not None else None,
    }


def stage_entry(entry, path: Path, token: str) -> dict:
    """Reuse only verified local bytes; download pinned sources atomically."""
    from explore_persona_space.orchestrate import hub

    if not path.exists():
        hub.stage_hub_file(
            REPO, entry.path, path, repo_type="dataset", revision=REVISION, token=token or None
        )
    return verify_hub_file(path, entry)


def namespace_for_group(group: str) -> str:
    """Translate the packing group into the corresponding capture-store name."""
    if group == "wildchat":
        return group
    kind, behavior = group.split("_", 1)
    if kind not in ("extraction", "labeling") or behavior not in BEHAVIORS:
        raise ValueError(f"unexpected raw-completion group: {group}")
    return f"{behavior}_{kind}"


def context_record(packed: dict, group: str) -> dict | None:
    """Read actual generation fields and reproduce capture's E1 row identities."""
    source = packed["src"]
    doc = packed["doc"]
    if not isinstance(source, str) or not isinstance(doc, dict):
        raise ValueError("packed record must carry a string src and object doc")
    if Path(source).name == "_manifest.json":
        return None
    namespace = namespace_for_group(group)
    if group.startswith("extraction_"):
        side = doc["sign"]
        if side not in ("pos", "neg"):
            raise ValueError(f"unexpected E1 sign in {source}")
        context_id = f"e1-pair{doc['pair']}-{side}-q{int(doc['q_idx']):02d}"
        question = doc["question"]
        rollouts = doc["rollouts"]
        if not isinstance(rollouts, list) or not rollouts:
            raise ValueError(f"empty E1 rollout list: {source}")
        rollout_ks = list(range(len(rollouts)))
        extraction = {"pair": doc["pair"], "side": side, "q_idx": int(doc["q_idx"])}
    else:
        context_id = doc["context_id"]
        question = doc["query"]
        rollout_k = doc["rollout_k"]
        if not isinstance(rollout_k, int) or rollout_k < 0:
            raise ValueError(f"invalid rollout index in {source}")
        rollout_ks = [rollout_k]
        extraction = {}
    if not isinstance(context_id, str) or not context_id:
        raise ValueError(f"missing context identity in {source}")
    full_exact, full_normalized = text_hashes(doc["prompt_text"])
    question_exact, question_normalized = text_hashes(question)
    turns = user_turn_hashes(doc["prompt_text"])
    meta = doc["meta"]
    generation = {key: meta[key] for key in ("model", "revision", "fingerprint", "git_commit")}
    if any(not isinstance(value, str) or not value for value in generation.values()):
        raise ValueError(f"incomplete generation provenance in {source}")
    return {
        "namespace": namespace,
        "context_id": context_id,
        "exact_sha256": full_exact,
        "normalized_sha256": full_normalized,
        "question_exact_sha256": question_exact,
        "question_normalized_sha256": question_normalized,
        "first_user_exact_sha256": turns[0]["exact_sha256"],
        "first_user_normalized_sha256": turns[0]["normalized_sha256"],
        "user_turn_hashes": turns,
        "split": doc.get("split"),
        "rung": doc.get("rung"),
        "group_key": doc.get("group_key"),
        **extraction,
        "generation_metadata": [generation],
        "rollouts": [
            {"rollout_k": k, "source_file": Path(source).name, "packed_src": source}
            for k in rollout_ks
        ],
    }


def merge_record(index: dict, record: dict) -> None:
    """Deduplicate context rows only when their prompts and memberships agree."""
    key = (record["namespace"], record["context_id"])
    if key not in index:
        index[key] = record
        return
    prior = index[key]
    variable = {"rollouts", "generation_metadata", "source_shards"}
    for field in set(prior) | set(record):
        if field not in variable and prior.get(field) != record.get(field):
            raise ValueError(f"inconsistent {field} for context {key}")
    seen_k = {row["rollout_k"] for row in prior["rollouts"]}
    for rollout in record["rollouts"]:
        if rollout["rollout_k"] in seen_k:
            raise ValueError(f"duplicate rollout identity for context {key}")
        prior["rollouts"].append(rollout)
        seen_k.add(rollout["rollout_k"])
    for generation in record["generation_metadata"]:
        if generation not in prior["generation_metadata"]:
            prior["generation_metadata"].append(generation)
    for shard in record.get("source_shards", []):
        if shard not in prior.setdefault("source_shards", []):
            prior["source_shards"].append(shard)


def read_group(paths: list[tuple[Path, dict]], group: str, group_meta: dict, index: dict) -> dict:
    """Check the full packing census while building the context index."""
    sources = set()
    manifest_rows = 0
    rollout_rows = 0
    for path, expected in paths:
        n_lines = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                packed = json.loads(line)
                source = packed["src"]
                if not source.startswith(group_meta["rel_dir"] + "/"):
                    raise ValueError(f"packed source outside its declared group: {group}/{source}")
                if source in sources:
                    raise ValueError(f"duplicate packed source: {group}/{source}")
                sources.add(source)
                n_lines += 1
                record = context_record(packed, group)
                if record is None:
                    manifest_rows += 1
                    continue
                record["source_shards"] = [path.name]
                rollout_rows += len(record["rollouts"])
                merge_record(index, record)
        if n_lines != expected["n_lines"]:
            raise ValueError(f"packing line-count mismatch: {path}")
    if len(sources) != group_meta["n_files"]:
        raise ValueError(f"packing file-count mismatch for {group}")
    return {
        "source_documents": len(sources),
        "manifest_documents": manifest_rows,
        "rollouts": rollout_rows,
    }


def stage_metadata(dest: Path, progress: Progress, token: str) -> dict:
    """Stage all seven namespaces and publish a complete provenance manifest."""
    records = {}
    sources = {}
    counts = {}
    completed = 0
    for label, prefix in PREFIXES.items():
        entries = {
            Path(e.path).name: e
            for e in scoped_entries(prefix, REVISION, token)
            if hasattr(e, "size")
        }
        if "pack_manifest.json" not in entries:
            raise FileNotFoundError(f"missing packing manifest at {prefix}")
        local = dest / "raw_metadata" / label
        local.mkdir(parents=True, exist_ok=True)
        manifest_path = local / "pack_manifest.json"
        info = stage_entry(entries["pack_manifest.json"], manifest_path, token)
        sources[str(manifest_path.relative_to(dest))] = info
        manifest = json.loads(manifest_path.read_text())
        expected_groups = (
            {"wildchat"}
            if label == "wildchat"
            else {
                f"{kind}_{behavior}"
                for kind in ("extraction", "labeling")
                for behavior in BEHAVIORS
            }
        )
        if set(manifest["groups"]) != expected_groups:
            raise ValueError(f"packing namespace set mismatch: {prefix}")
        expected_shards = {s["name"] for g in manifest["groups"].values() for s in g["shards"]}
        realized_shards = {name for name in entries if re.fullmatch(r".+\.shard\d+\.jsonl", name)}
        if expected_shards != realized_shards:
            raise ValueError(f"packing shard set disagrees with complete Hub listing: {prefix}")
        for group, group_meta in sorted(manifest["groups"].items()):
            if group_meta["n_shards"] != len(group_meta["shards"]):
                raise ValueError(f"packing shard count mismatch for {group}")
            paths = []
            for expected in group_meta["shards"]:
                progress.check()
                name = expected["name"]
                path = local / name
                progress.emit(phase="stage_prompt_metadata", group=group, latest_member=name)
                info = stage_entry(entries[name], path, token)
                if not verified_file(path, expected):
                    raise RuntimeError(f"missing downloaded metadata source: {path}")
                sources[str(path.relative_to(dest))] = info
                paths.append((path, expected))
                completed += 1
                progress.emit(
                    files_completed=completed,
                    bytes_fetched=sum(r["bytes"] for r in sources.values()),
                )
            counts[namespace_for_group(group)] = read_group(paths, group, group_meta, records)
            progress.emit(
                phase="indexed_prompt_metadata", group=group, contexts_indexed=len(records)
            )
    entries = scoped_entries(str(Path(JUDGE_PATH).parent), REVISION, token)
    matched = [e for e in entries if e.path == JUDGE_PATH]
    if len(matched) != 1:
        raise FileNotFoundError(f"pinned per-rollout judge source missing: {JUDGE_PATH}")
    judge_path = dest / "provenance" / "hallucination_per_rollout.json"
    info = stage_entry(matched[0], judge_path, token)
    sources[str(judge_path.relative_to(dest))] = info
    provenance = dest / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    index_path = provenance / "prompt_index.jsonl"
    temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as stream:
        for key, record in sorted(records.items()):
            record["rollouts"].sort(key=lambda r: r["rollout_k"])
            record["source_shards"].sort()
            record["generation_metadata"].sort(key=lambda r: json.dumps(r, sort_keys=True))
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            counts[key[0]]["contexts"] = counts[key[0]].get("contexts", 0) + 1
    temporary.replace(index_path)
    result = {
        "repo": REPO,
        "revision": REVISION,
        "hash_recipe": HASH_RECIPE,
        "sources": sources,
        "namespaces": counts,
        "prompt_index": {
            "path": str(index_path.relative_to(dest)),
            "sha256": sha256(index_path),
            "bytes": index_path.stat().st_size,
            "contexts": len(records),
        },
        "complete": True,
    }
    write_json(provenance / "metadata_manifest.json", result)
    progress.emit(
        phase="complete",
        files_completed=len(sources),
        contexts_indexed=len(records),
        manifest=str(provenance / "metadata_manifest.json"),
    )
    return result


def main() -> None:
    """Run only the pinned metadata stage; model inference is never invoked."""
    from explore_persona_space.orchestrate.env import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    args.dest.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.progress)
    try:
        stage_metadata(args.dest, progress, token)
    finally:
        progress.close()


if __name__ == "__main__":
    main()
