"""Stage pinned draw-zero responses and audit query matching for the K5 cohort."""

# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k5_loso_calibration as base

PARENT_REV = "5ae90722bf11330deddfa42cf41f9fec6da8b69f"
MANIFEST = "issue2054_section44_k3_gcp/capture_recovery_v1/manifest.json"


def fetch(path, revision):
    """Fetch an immutable input through the existing transport-retry helper."""
    return Path(
        base.retry_transient(
            lambda: hf_hub_download(base.HF_REPO, path, repo_type="dataset", revision=revision),
            what=f"matched-response input {path}",
        )
    )


def stage(out, inputs):
    """Verify all original responses that supply draw zero of the K5 aggregates."""
    manifest_path = fetch(MANIFEST, PARENT_REV)
    manifest = json.loads(manifest_path.read_text())
    sources = [r for r in json.loads(inputs.read_text()) if "/production_v1/k5/" in r["path"]]
    cells = {Path(r["path"]).stem: r for r in sources}
    records = [r for r in manifest["cells"] if r["cell"] in cells]
    if len(records) != 12:
        raise RuntimeError("expected twelve K5 raw sources")

    def load(record):
        path = fetch(record["raw"], manifest["revision"])
        if base.sha(path) != record["raw_sha256"]:
            raise RuntimeError("original raw response hash mismatch")
        with path.open() as handle:
            rows = [json.loads(line) for line in handle]
        if len(rows) != record["n"] or len({r["conv_id"] for r in rows}) != len(rows):
            raise RuntimeError("raw row count or unique identity mismatch")
        source = cells[record["cell"]]
        if base.sha(source["local"]) != source["sha256"]:
            raise RuntimeError("K5 bank changed")
        with np.load(source["local"], allow_pickle=False) as z:
            ids = set(map(str, z["conv_id"]))
        selected = {r["conv_id"]: r for r in rows if r["conv_id"] in ids}
        if set(selected) != ids:
            raise RuntimeError("K5 cohort missing original responses")
        for row in selected.values():
            if row["final_text"][row["answer_start"] : row["answer_end"]] != row["answer"]:
                raise RuntimeError("stored answer boundary mismatch")
        reference = {
            "cell": record["cell"],
            "revision": manifest["revision"],
            "path": record["raw"],
            "local": str(path),
            "sha256": base.sha(path),
            "raw_rows": len(rows),
            "k5_rows": len(selected),
        }
        print(f"[phase=raw_verified] {record['cell']} rows={len(selected)}", flush=True)
        return reference, selected

    with ThreadPoolExecutor(max_workers=4) as executor:
        loaded = list(executor.map(load, records))
    out.mkdir(parents=True, exist_ok=True)
    base.atomic_json(
        out / "raw_inputs.json",
        {
            "manifest_revision": PARENT_REV,
            "manifest_path": MANIFEST,
            "manifest_sha256": base.sha(manifest_path),
            "sources": [ref for ref, _ in loaded],
        },
    )
    data = {ref["cell"]: rows for ref, rows in loaded}
    audits = []
    for model in base.MODELS:
        chat = data[f"{base.SETTINGS[0][1]}__{model}"]
        queries = {}
        for cid, row in chat.items():
            prefix = row["final_text"][: row["answer_start"]]
            before, sep, tail = prefix.rpartition("<|im_start|>user\n")
            if not sep or not tail.endswith("<|im_end|>\n<|im_start|>assistant\n"):
                raise RuntimeError(f"unexpected chat prompt layout {cid}: {before[:80]}")
            queries[cid] = tail.removesuffix("<|im_end|>\n<|im_start|>assistant\n")
        for label, prefix in base.SETTINGS:
            cell = f"{prefix}__{model}"
            common = sorted(chat.keys() & data[cell].keys())
            exact, normalized, missing = [], [], []
            for cid in common:
                r = data[cell][cid]
                context = r["final_text"][: r["answer_start"]]
                query = queries[cid]
                if query in context:
                    exact.append(cid)
                elif " ".join(query.split()) in " ".join(context.split()):
                    normalized.append(cid)
                else:
                    missing.append(cid)
            audits.append(
                {
                    "cell": cell,
                    "n_matched_to_chat": len(common),
                    "query_exact_present": len(exact),
                    "query_whitespace_normalized_present": len(normalized),
                    "missing_ids": missing,
                }
            )
            print(
                f"[phase=query_audit] {model} {label!r} exact={len(exact)} whitespace={len(normalized)} missing={len(missing)}",
                flush=True,
            )
        base.atomic_json(out / f"queries_{model}.json", queries)
    base.atomic_json(
        out / "query_audit.json",
        {
            "audit": audits,
            "definition": "Same conv_id, and the full assistant-chat user query appears before the answer boundary in the other framing; exact and whitespace-only matches are distinguished.",
        },
    )


def main():
    """Stage the bounded response corpus for direct qualitative inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    stage(args.out, args.inputs)


if __name__ == "__main__":
    main()
