"""Repeat the offset test only where the full canonical query appears in both prompts."""

# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k5_loso_calibration as base
from scripts import issue2054_k5_matched_offsets as core


def validated_sources(inputs, parent):
    """Bind the completed broad analysis, current banks, and raw provenance."""
    sources = {
        r["path"]: r for r in json.loads(inputs.read_text()) if "/production_v1/k5/" in r["path"]
    }
    broad = json.loads((parent / "results.json").read_text())
    for row in broad["pairs"]:
        expected = [
            sources[f"{base.PREFIX}/production_v1/k5/{c}.npz"]["sha256"]
            for c in (row["source"], row["target"])
        ]
        if row["input_sha256"] != expected:
            raise RuntimeError("broad results do not match current input identity")
    for ref in sources.values():
        if base.sha(ref["local"]) != ref["sha256"]:
            raise RuntimeError("bank changed")
    return sources


def fit(out, inputs, parent, model):
    """Learn and score the same constant shift after literal query auditing."""
    sources = validated_sources(inputs, parent)
    panel = base.load_panel(model, sources)
    queries = json.loads((parent / f"queries_{model}.json").read_text())
    refs = json.loads((parent / "raw_inputs.json").read_text())["sources"]
    valid = {}
    for ref in refs:
        cell = ref["cell"]
        if cell not in panel:
            continue
        if base.sha(ref["local"]) != ref["sha256"]:
            raise RuntimeError("raw input changed")
        with Path(ref["local"]).open() as handle:
            rows = {r["conv_id"]: r for r in map(json.loads, handle)}
        valid[cell] = {
            cid
            for cid in panel[cell]["ids"]
            if cid in queries
            and " ".join(queries[cid].split())
            in " ".join(rows[cid]["final_text"][: rows[cid]["answer_start"]].split())
        }
    prefixes = [p for _, p in base.SETTINGS]
    for i, j in itertools.combinations(range(6), 2):
        ca, cb = [f"{prefixes[k]}__{model}" for k in (i, j)]
        a, b = panel[ca], panel[cb]
        ai, bi = {c: n for n, c in enumerate(a["ids"])}, {c: n for n, c in enumerate(b["ids"])}
        ids = sorted(valid[ca] & valid[cb])
        if not ids:
            raise RuntimeError("empty literal-query intersection")
        ia, ib = np.array([ai[c] for c in ids]), np.array([bi[c] for c in ids])
        folds = a["membership"][ia]
        if not np.array_equal(folds, b["membership"][ib]):
            raise RuntimeError("fold mismatch")
        for arm, key in [("context", "x"), ("answer", "y")]:
            path = out / "pairs" / f"{model}__{i}_{j}__{arm}.json"
            if path.exists():
                raise RuntimeError("strict run already has output; use a fresh output directory")
            records, arrays = core.evaluate_pair(a[key][ia], b[key][ib], folds)
            arrays["conv_id"] = np.array(ids)
            arrays["source_cap_mask"], arrays["target_cap_mask"] = a["caps"][ia], b["caps"][ib]
            base.save_npz(path.with_suffix(".npz"), arrays)
            record = {
                "model": model,
                "source": ca,
                "target": cb,
                "source_index": i,
                "target_index": j,
                "arm": arm,
                "n_paired": len(ids),
                "n_shared_id": len(ai.keys() & bi.keys()),
                "n_shared_id_with_chat_query": len(ai.keys() & bi.keys() & queries.keys()),
                "constant_fraction": 1
                - float(arrays["residual_energy"].sum() / arrays["displacement_energy"].sum()),
                "r2_mean": {
                    name: float(np.mean([r["metrics"][name]["r2"] for r in records]))
                    for name in ("identity", "bias")
                },
                "top1_mean": {
                    name: float(np.mean([r["metrics"][name]["euclidean_top1"] for r in records]))
                    for name in ("identity", "bias")
                },
                "folds": records,
                "core_sha256": base.sha(core.__file__),
                "script_sha256": base.sha(__file__),
                "array_sha256": base.sha(path.with_suffix(".npz")),
                "input_sha256": [
                    sources[f"{base.PREFIX}/production_v1/k5/{c}.npz"]["sha256"] for c in (ca, cb)
                ],
                "raw_inputs_sha256": base.sha(parent / "raw_inputs.json"),
                "queries_sha256": base.sha(parent / f"queries_{model}.json"),
            }
            base.atomic_json(path, record)
            print(
                f"[phase=pair] {model} {i}->{j} {arm} literal_n={len(ids)} fraction={record['constant_fraction']:.4f}",
                flush=True,
            )


def collect(out, inputs, parent):
    """Validate all strict results, their current input hashes, and full coverage."""
    sources = validated_sources(inputs, parent)
    rows = []
    for model in base.MODELS:
        for i, j in itertools.combinations(range(6), 2):
            for arm in ("context", "answer"):
                path = out / "pairs" / f"{model}__{i}_{j}__{arm}.json"
                row = json.loads(path.read_text())
                expected = [
                    sources[f"{base.PREFIX}/production_v1/k5/{c}.npz"]["sha256"]
                    for c in (row["source"], row["target"])
                ]
                checks = [
                    row["input_sha256"] == expected,
                    row["array_sha256"] == base.sha(path.with_suffix(".npz")),
                    row["core_sha256"] == base.sha(core.__file__),
                    row["script_sha256"] == base.sha(__file__),
                    row["raw_inputs_sha256"] == base.sha(parent / "raw_inputs.json"),
                    row["queries_sha256"] == base.sha(parent / f"queries_{model}.json"),
                ]
                if not all(checks):
                    raise RuntimeError("strict checkpoint provenance changed")
                rows.append(row)
    broad = json.loads((parent / "results.json").read_text())
    broad.update(
        pairs=rows,
        cohort="Same conversation ID and full canonical assistant-chat query present in both prefixes, allowing whitespace differences only; conversation must occur in the chat bank.",
        broad_parent_results_sha256=base.sha(parent / "results.json"),
        validation="All bank, raw-source-manifest, query-file, inference-code and checkpoint hashes verified. No resume used for this strict run.",
    )
    base.atomic_json(out / "results.json", broad)
    base.atomic_json(out / "inputs.json", list(sources.values()))


def main():
    """Fit one checkpoint or validate and collect the complete strict analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["fit", "collect"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", choices=base.MODELS)
    args = parser.parse_args()
    parent = args.out.parent / "k5_matched_offsets"
    if args.stage == "fit":
        fit(args.out, args.inputs, parent, args.model)
    else:
        collect(args.out, args.inputs, parent)


if __name__ == "__main__":
    main()
