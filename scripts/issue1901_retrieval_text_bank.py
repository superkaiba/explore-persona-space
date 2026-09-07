#!/usr/bin/env python3
"""Join verified prompt and stored response text to the fixed 10k candidate bank.

The required ``--round1-prompts`` JSON contains the original 5,000 prompt strings.
Its ordered, null-separated prompt SHA must match the pinned bundle metadata;
every selected prompt is additionally checked against the pinned bundle index.
One known re-derivation route is ``issue779_ffc_n50k_generate_capture.
sample_disjoint_n50k(5000, 0, 0)['round1']`` with authorized access to LMSYS.
That route is not run here: the cache is an explicit input, never silently rebuilt.

Responses are existing seed-43 draws for every candidate and seeds 44–46 for test
candidates. The four distractor-generation shards are strided; each chunk has
500 rows. These are the producing issue1901_avgpool_scaleup.py conventions.
No generation or fitting occurs. Source files and their Hub hashes are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
REVISION = "83d249cc9d495ca6f5d10f9156a622bcdca29a19"
BASE = "issue1901_avgpool"
MANIFEST = "issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest"
LOG = logging.getLogger("retrieval_text_bank")


def sha256(path: Path) -> str:
    """Hash a required small text input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_files(names: list[str], args: argparse.Namespace) -> dict:
    """Stage only named files and verify local bytes against their pinned Hub blobs."""
    api = HfApi()
    entries = hub.retry_transient(
        lambda: api.get_paths_info(REPO, names, repo_type="dataset", revision=REVISION),
        what="retrieval text source metadata",
    )
    infos = {r.path: r for r in entries}
    if set(infos) != set(names):
        raise RuntimeError(f"missing pinned text sources: {set(names) - set(infos)}")

    def one(name: str) -> tuple[str, dict]:
        """Copy a cached source or download it, then verify the exact content."""
        destination = args.stage_root / name
        cached = args.source_cache / name if args.source_cache is not None else None
        if not destination.exists() and cached is not None and cached.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, destination)
        elif not destination.is_file():
            destination = Path(
                hub.retry_transient(
                    lambda: hf_hub_download(
                        REPO,
                        filename=name,
                        repo_type="dataset",
                        revision=REVISION,
                        local_dir=args.stage_root,
                    ),
                    what=f"retrieval text source {name}",
                )
            )
        content = destination.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        info = infos[name]
        if info.lfs is not None:
            actual, expected = digest, info.lfs.sha256
        else:
            actual = hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest()
            expected = info.blob_id
        if actual != expected:
            raise RuntimeError(f"pinned text content mismatch: {name}")
        return name, {
            "local_path": str(destination),
            "repo_path": name,
            "revision": REVISION,
            "sha256": digest,
            "bytes": len(content),
            "verified_against_pinned_hub": True,
        }

    with ThreadPoolExecutor(max_workers=4) as workers:
        result = dict(workers.map(one, names))
    LOG.info("verified %d text sources", len(result))
    return result


def assemble(args: argparse.Namespace) -> Path:
    """Build all candidate text rows with capture-ID and prompt-hash checks."""
    pool = json.loads(args.pool.read_text())
    wanted = set(pool["pool_capture_ids"])
    if len(wanted) != 10000 or len(pool["pool_capture_ids"]) != 10000:
        raise ValueError("the text bank requires the exact 10k candidate pool")
    sources = stage_files(
        [f"{BASE}/analysis_tensors/bundle/bundle_{suffix}.json" for suffix in ("index", "meta")],
        args,
    )
    index_doc = json.loads(
        (args.stage_root / BASE / "analysis_tensors/bundle/bundle_index.json").read_text()
    )
    meta = json.loads(
        (args.stage_root / BASE / "analysis_tensors/bundle/bundle_meta.json").read_text()
    )
    index = {r["ci"]: r for r in index_doc["rows"]}
    if len(index) != len(index_doc["rows"]):
        raise RuntimeError("duplicate capture IDs in bundle index")
    round1 = json.loads(args.round1_prompts.read_text())
    if len(round1) != 5000 or not all(isinstance(p, str) for p in round1):
        raise ValueError("round1 input must contain 5,000 prompt strings")
    ordered_sha = hashlib.sha256()
    for prompt in round1:
        ordered_sha.update(prompt.encode() + b"\0")
    if ordered_sha.hexdigest() != meta["round1_sha256"]:
        raise RuntimeError("round1 ordered prompt hash does not match pinned metadata")
    local_round1 = args.stage_root / "first5000_prompts.json"
    if local_round1.exists():
        if sha256(local_round1) != sha256(args.round1_prompts):
            raise RuntimeError("existing round1 JSON bytes differ from the explicit input")
    elif local_round1 != args.round1_prompts:
        shutil.copyfile(args.round1_prompts, local_round1)
    texts = {}
    for eval_row, passb_row in enumerate(pool["test_rows"]):
        ci = -(eval_row + 1)
        prompt = round1[passb_row]
        if hashlib.sha256(prompt.encode()).hexdigest() != index[ci]["prompt_sha256"]:
            raise RuntimeError(f"test prompt hash mismatch at capture ID {ci}")
        if ci in wanted:
            texts[ci] = {
                "ci": ci,
                "source_type": "test",
                "corpus": "lmsys",
                "eval_row": eval_row,
                "passb_row": passb_row,
                "prompt": prompt,
                "prompt_sha256": index[ci]["prompt_sha256"],
                "prompt_source": str(local_round1),
                "answers": {},
            }
    part = 0
    while set(texts) != wanted:
        name = f"{MANIFEST}/part_{part:05d}.jsonl"
        sources.update(stage_files([name], args))
        with (args.stage_root / name).open() as handle:
            # File iteration preserves literal Unicode line separators inside JSON strings.
            for line in handle:
                row = json.loads(line)
                ci = row["i"]
                if ci not in wanted:
                    continue
                if ci in texts:
                    raise RuntimeError(f"duplicate manifest capture ID {ci}")
                if hashlib.sha256(row["prompt"].encode()).hexdigest() != index[ci]["prompt_sha256"]:
                    raise RuntimeError(f"distractor prompt hash mismatch at capture ID {ci}")
                texts[ci] = {
                    "ci": ci,
                    "source_type": "distr",
                    "corpus": row["corpus"],
                    "stream_pos": row["stream_pos"],
                    "prompt": row["prompt"],
                    "prompt_sha256": index[ci]["prompt_sha256"],
                    "prompt_source": name,
                    "answers": {},
                }
        part += 1
        LOG.info("manifest parts=%d selected prompts=%d/10000", part, len(texts))

    locations = {}
    for kind in ("test", "distr"):
        rows = [r for r in index_doc["rows"] if r["src"] == kind]
        shards = 1 if kind == "test" else 4
        for position, row in enumerate(rows):
            if row["ci"] not in wanted:
                continue
            shard, chunk = position % shards, (position // shards) // 500
            seeds = (43, 44, 45, 46) if kind == "test" else (43,)
            for seed in seeds:
                name = f"{BASE}/raw_completions/{kind}/shard{shard:02d}/gen_seed{seed}_chunk{chunk}.json"
                locations[name] = (seed, kind, shard, chunk)
    disclosure = f"{BASE}/raw_completions/distr/shard00/scrub_disclosures_seed43.json"
    sources.update(stage_files([*sorted(locations), disclosure], args))
    for name, (seed, kind, shard, chunk) in sorted(locations.items()):
        doc = json.loads((args.stage_root / name).read_text())
        if doc["meta"]["bundle_sha"] != index_doc["ci_sha256"]:
            raise RuntimeError(f"answer bundle fingerprint mismatch: {name}")
        if (doc["meta"]["seed"], doc["meta"]["shard"], doc["meta"]["chunk"]) != (
            seed,
            shard,
            chunk,
        ):
            raise RuntimeError(f"answer seed/shard/chunk mismatch: {name}")
        for row in doc["rows"]:
            ci = row["ci"]
            if row["src"] != kind or index[ci]["row_idx"] != row["row_idx"]:
                raise RuntimeError(f"answer row identity mismatch: {name}, {ci}")
            if ci not in wanted:
                continue
            if str(seed) in texts[ci]["answers"]:
                raise RuntimeError(f"duplicate answer: {ci}, seed {seed}")
            texts[ci]["answers"][str(seed)] = {
                "response": row["response"],
                "source": name,
                "seed": seed,
                "response_sha256": hashlib.sha256(row["response"].encode()).hexdigest(),
                "prompt_token_ids_sha256": hashlib.sha256(
                    json.dumps(row["prompt_token_ids"], separators=(",", ":")).encode()
                ).hexdigest(),
            }
    coverage = {
        str(seed): sum(str(seed) in row["answers"] for row in texts.values())
        for seed in range(43, 47)
    }
    if coverage != {"43": 10000, "44": 942, "45": 942, "46": 942}:
        raise RuntimeError(f"incomplete stored response coverage: {coverage}")
    bank = {
        "schema_version": 1,
        "repo_id": REPO,
        "data_revision": REVISION,
        "pool_source": str(args.pool),
        "pool_sha256": sha256(args.pool),
        "bundle_ci_sha256": index_doc["ci_sha256"],
        "bundle_meta": meta,
        "coverage": {
            "n_candidates": 10000,
            "n_prompts": len(texts),
            "response_coverage_by_seed": coverage,
            "test_prompts_sha_verified": 1000,
            "selected_prompts_sha_verified": 10000,
        },
        "sources": sources,
        "round1_local_source": {
            "path": str(local_round1),
            "sha256": sha256(local_round1),
            "ordered_prompt_sha256": ordered_sha.hexdigest(),
            "validation": "Ordered prompt hash and all 1,000 test hashes match pinned metadata.",
        },
        "scrub_disclosure_sources": [disclosure],
        "provenance_notes": [
            "Stored seed43 is representative text; retrieval uses five-answer-vector means.",
            "No text generation or fitting; all source bytes are checked at the pinned revision.",
            "Test capture ID is -(eval_row+1), where eval_row precedes deduplication.",
        ],
        "rows": {str(ci): texts[ci] for ci in pool["pool_capture_ids"]},
    }
    output = args.output if args.output is not None else args.stage_root / "text_bank.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(bank, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)
    LOG.info("wrote %s; coverage=%s", output, coverage)
    return output


def main() -> None:
    """Parse the explicit local prompt cache and optional pinned-source cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1-prompts", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument(
        "--stage-root", type=Path, default=ROOT / "data/issue_1901/retrieval_10k/text_sources"
    )
    parser.add_argument(
        "--pool", type=Path, default=ROOT / "eval_results/issue_1901/retrieval_10k/pool.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    assemble(args)


if __name__ == "__main__":
    main()
