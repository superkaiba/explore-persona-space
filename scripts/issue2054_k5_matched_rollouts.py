"""Retrieve all five existing draws for three safe illustrative matched queries."""

# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k5_matched_responses as raw

IDS = ("stripped_s3789", "stripped_s3263", "stripped_s1099")


def main():
    """Verify raw chunk receipts and answer/prompt identity for the selected draws."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parent = args.out.parent / "k5_matched_offsets"
    references = json.loads((parent / "raw_inputs.json").read_text())["sources"]
    banks = {Path(r["path"]).stem: r for r in json.loads((parent / "inputs.json").read_text())}
    jobs, originals, sources = [], {}, []
    for ref in references:
        cell = ref["cell"]
        if not cell.endswith("__qwen2.5-7b-instruct"):
            continue
        if raw.base.sha(ref["local"]) != ref["sha256"]:
            raise RuntimeError("original response changed")
        with Path(ref["local"]).open() as handle:
            rows = list(map(json.loads, handle))
        with np.load(banks[cell]["local"], allow_pickle=False) as bank:
            valid_ids = set(map(str, bank["conv_id"]))
        chosen = [
            (i, r) for i, r in enumerate(rows) if r["conv_id"] in IDS and r["conv_id"] in valid_ids
        ]
        for _, r in chosen:
            originals[(cell, r["conv_id"])] = r
        for offset in sorted({i // 256 * 256 for i, _ in chosen}):
            for revision, prefix, draws in [
                (raw.PARENT_REV, "issue2054_section44_k3_gcp/production", (1, 2)),
                (raw.base.K5_REV, "issue2054_section44_k5_gcp/production_v1", (3, 4)),
            ]:
                jobs.append((cell, offset, revision, prefix, draws))

    def verified(path, revision):
        local = raw.fetch(path, revision)
        receipt = json.loads(raw.fetch(path + ".done.json", revision).read_text())
        digest = raw.base.sha(local)
        if digest != receipt["sha256"] or local.stat().st_size != receipt["size"]:
            raise RuntimeError("raw chunk receipt mismatch")
        return local, {"revision": revision, "path": path, "sha256": digest, "receipt": receipt}

    def load(job):
        cell, offset, revision, prefix, draws = job
        index_path = f"{prefix}/raw/{cell}/chunk_{offset:05d}.json"
        index, ref = verified(index_path, revision)
        info = json.loads(index.read_text())
        records, provenance = [], [ref]
        for shard in info["shards"]:
            path, reference = verified(str(Path(index_path).parent / shard), revision)
            records.extend(json.loads(path.read_text()))
            provenance.append(reference)
        if len(records) != info["rows"]:
            raise RuntimeError("raw chunk row count mismatch")
        selected = []
        for row in records:
            key = (cell, row["conv_id"])
            if key not in originals:
                continue
            orig = originals[key]
            if (
                row["draw"] not in draws
                or row["final_text"][: row["answer_start"]]
                != orig["final_text"][: orig["answer_start"]]
            ):
                raise RuntimeError("draw or prompt identity mismatch")
            if row["final_text"][row["answer_start"] : row["answer_end"]] != row["answer"]:
                raise RuntimeError("answer boundary mismatch")
            selected.append(
                {"cell": cell, "row": row, "source_path": index_path, "source_revision": revision}
            )
        print(
            f"[phase=rollouts] {cell} offset={offset} draws={draws} selected={len(selected)}",
            flush=True,
        )
        return selected, provenance

    selected = [
        {
            "cell": cell,
            "row": dict(row, draw=0),
            "source_revision": next(r["revision"] for r in references if r["cell"] == cell),
            "source_path": next(r["path"] for r in references if r["cell"] == cell),
        }
        for (cell, _), row in originals.items()
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for rows, provenance in executor.map(load, jobs):
            selected.extend(rows)
            sources.extend(provenance)
    for cell, cid in originals:
        draws = [
            r["row"]["draw"] for r in selected if r["cell"] == cell and r["row"]["conv_id"] == cid
        ]
        if sorted(draws) != list(range(5)):
            raise RuntimeError("selected example lacks exactly five verified draws")
    raw.base.atomic_json(
        args.out / "selected_rollouts.json",
        {
            "selection": "Three illustrative queries chosen after reading draw zero: CSV-link ability, free will, date. Exploratory examples, not a random behavioral sample.",
            "ids": IDS,
            "rows": selected,
            "sources": sources,
            "original_sources": references,
            "verified_all_five_draws": True,
        },
    )
    print(
        f"[phase=rollouts_complete] {len(selected)} saved responses; no new generation", flush=True
    )


if __name__ == "__main__":
    main()
