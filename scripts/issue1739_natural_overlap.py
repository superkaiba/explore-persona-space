"""Re-score saved #1739 predictions after the registered map-content exclusion.

Usage: issue1739_natural_overlap.py RUN_ROOT [--output PATH]
Only small pinned corpus text is fetched. Maps, directions, labels and captured
responses remain fixed. Every mask, bootstrap draw and consumed hash is saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.experiments.issue_1739 import arms, store_io  # noqa: E402
from explore_persona_space.experiments.issue_1739.corpus_staging import (  # noqa: E402
    minhash_signatures,
    near_dup_mask,
)
from explore_persona_space.orchestrate import hub  # noqa: E402
from scripts.issue1739_claim4_fold import (  # noqa: E402
    REGISTERED_PRIMARY_RUNGS,
    group_bootstrap_rhos,
)
from scripts.issue1739_natural_score import LAYERS, sha, write_json  # noqa: E402

BEHAVIORS = ("sycophancy", "hallucination", "evil")
METHODS = tuple(LAYERS["sycophancy"])
SERIES = tuple(f"{v}/{m}" for v in ("e1", "q01_s0") for m in METHODS)
CORPUS_REVISION = "7ef5523673d64697ab497577dbc5b9270c39f020"
CORPUS_PREFIX = "issue1092_realistic_crossing/corpus"
CORPUS_BLOBS = {
    "manifest.jsonl": "4411f7ab5f5233db56b0ef94cb43b8a60a7fe8c2",
    "query_store.jsonl": "7e71602b7a111e6e3e89b01269f576299b61f1b3",
    "prefix_store.jsonl": "bab87051a5f302591edb7dbfef9d4d0199972bed",
}
RECIPE = {
    "normalization": "lowercase and collapse whitespace",
    "generic_exact": "evaluation query/any user turn versus fit queries/any prefix user turn",
    "trait_exact": "evaluation query/any user turn versus trait-train query/any user turn",
    "near": "evaluation query versus generic query/user-turn or trait-train query",
    "near_recipe": "corpus_staging MinHash char5,64 permutations,seed0,16 bands x4",
    "limitation": "Operational exact/LSH disjointness; not semantic or paraphrase certification",
    "bootstrap": "500 paired source-group draws within each dataset; equal-dataset OOD mean",
    "conditioning": "Cached responses, judged labels, extraction directions and fitted maps fixed",
    "id": "unavailable: every original ID evaluation row was exposed to the trait map fit",
}


def require(condition, message):
    """Fail loudly on an incomplete join, changed input or invalid estimate."""
    if not condition:
        raise ValueError(message)


def record(path, provenance, expected=None):
    """Record a consumed file after enforcing an available immutable SHA256."""
    digest = sha(path)
    require(expected is None or digest == expected, f"Input hash mismatch: {path}")
    provenance[str(Path(path).resolve())] = digest
    return digest


def read_json(path, provenance, expected=None):
    """Read JSON only after registering its exact consumed bytes."""
    record(path, provenance, expected)
    return json.loads(Path(path).read_text())


def normalized_hash(text):
    """Apply the frozen natural-prompt lowercase/whitespace hash recipe."""
    require(isinstance(text, str) and bool(text.strip()), "Empty/non-string prompt")
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def atomic_npz(path, **arrays):
    """Publish a complete NumPy artifact without exposing partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def signatures(keys, texts, target, provenance):
    """Cache bounded metadata-only chunks with full-content/recipe identity."""
    identity = hashlib.sha256(
        json.dumps(
            [
                RECIPE["near_recipe"],
                sha(Path(minhash_signatures.__code__.co_filename)),
                list(zip(keys, map(normalized_hash, texts))),
            ]
        ).encode()
    ).hexdigest()
    arrays = []
    for start in range(0, len(keys), 128):
        path = target / f"signatures_{start:06d}.npz"
        ck = keys[start : start + 128]
        if path.exists():
            with np.load(path) as saved:
                require(str(saved["fingerprint"]) == identity, f"Stale signature chunk: {path}")
                require(saved["ids"].tolist() == ck, f"Wrong signature IDs: {path}")
                array = saved["signatures"]
                require(array.shape == (len(ck), 64), f"Invalid signature shape: {path}")
        else:
            array = minhash_signatures(texts[start : start + 128], seed=0)
            atomic_npz(path, fingerprint=identity, ids=np.asarray(ck), signatures=array)
        arrays.append(array)
        record(path, provenance)
        print(f"[signatures] {target.name} {min(start + 128, len(keys))}/{len(keys)}", flush=True)
    return np.concatenate(arrays)


def generic_pool(out, provenance):
    """Join every pinned generic fit row to its exact query/prefix source text."""
    stores = {}
    for name, blob in CORPUS_BLOBS.items():
        path = out / "generic_sources" / name
        hub.stage_hub_file(
            hub.DEFAULT_DATASET_REPO,
            f"{CORPUS_PREFIX}/{name}",
            path,
            repo_type="dataset",
            revision=CORPUS_REVISION,
        )
        data = path.read_bytes()
        require(
            hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == blob,
            f"Pinned corpus Git blob mismatch: {name}",
        )
        record(path, provenance)
        stores[name] = [json.loads(line) for line in data.splitlines() if line.strip()]
    rows = stores["manifest.jsonl"]
    fit = [r for r, keep in zip(rows, store_io.fit_pool_mask(rows), strict=True) if keep]
    require(len(fit) == 18793, "Wrong generic map-fit row count")
    query = {r["query_id"]: r for r in stores["query_store.jsonl"]}
    prefix = {r["prefix_id"]: r for r in stores["prefix_store.jsonl"]}
    qids, pids = {r["query_id"] for r in fit}, {r["prefix_id"] for r in fit}
    require(qids <= query.keys() and pids <= prefix.keys(), "Incomplete generic text join")
    texts = [query[q]["text"] for q in sorted(qids)] + [
        turn["content"]
        for p in sorted(pids)
        for turn in prefix[p]["prefix_turns"]
        if turn["role"] == "user" and turn["content"].strip()
    ]
    unique = {normalized_hash(t): t for t in texts}
    keys = sorted(unique)
    sigs = signatures(keys, [unique[k] for k in keys], out / "signatures/generic", provenance)
    coverage = {
        "revision": CORPUS_REVISION,
        "prefix": CORPUS_PREFIX,
        "git_blobs": CORPUS_BLOBS,
        "fit_rows": len(fit),
        "query_ids": len(qids),
        "prefix_ids": len(pids),
        "joined_query_ids": len(qids & query.keys()),
        "joined_prefix_ids": len(pids & prefix.keys()),
        "unique_query_or_userturn_texts": len(unique),
    }
    write_json(out / "generic_coverage.json", coverage)
    return set(unique), sigs, coverage


def load_contexts(run, behavior, config, out, provenance):
    """Verify compact input shards and reconstruct only prompt metadata hashes."""
    source = run / "inputs" / behavior
    metadata = read_json(
        source / "metadata.json",
        provenance,
        config["behaviors"][behavior]["hf_files"]["prompts.jsonl"]["sha256"],
    )
    contexts = {}
    for part in metadata["context_parts"]:
        rows = read_json(source / part["path"], provenance, part["sha256"])
        require(len(rows) == part["n"] and not contexts.keys() & rows.keys(), "Bad context shard")
        contexts.update(rows)
    require(len(contexts) == metadata["n_contexts"], "Incomplete context shard union")
    keys = sorted(contexts)
    for key in keys:
        require(
            normalized_hash(contexts[key]["query"]) == contexts[key]["question_normalized_sha256"],
            f"Query hash mismatch: {key}",
        )
    sigs = signatures(
        keys, [contexts[k]["query"] for k in keys], out / "signatures" / behavior, provenance
    )
    return contexts, dict(zip(keys, sigs, strict=True))


def load_predictions(root, behavior, dataset, fetch, provenance):
    """Pair exact IDs, groups and DVs before choosing each frozen readout layer."""
    anchor, selected = None, {}
    for layer in sorted(set(LAYERS[behavior].values())):
        name = f"L{layer:02d}/predictions_{dataset}.npz"
        path = root / name
        record(path, provenance, fetch["files"][name]["sha256"])
        with np.load(path) as archive:
            data = {k: archive[k] for k in archive.files}
        if anchor is None:
            anchor = data
        else:
            require(
                all(np.array_equal(anchor[k], data[k]) for k in ("ids", "groups", "dv")),
                f"Evaluation rows differ between layers: {dataset}",
            )
        for i, (variant, method) in enumerate(zip(data["variants"], data["arms"], strict=True)):
            key = f"{variant}/{method}"
            if key in SERIES and layer == LAYERS[behavior][method]:
                require(key not in selected, f"Duplicate selected readout: {key}")
                selected[key] = data["predictions"][i]
    require(set(selected) == set(SERIES), f"Missing primary readout: {behavior}/{dataset}")
    matrix = np.asarray([selected[k] for k in SERIES])
    require(np.isfinite(matrix).all() and np.isfinite(anchor["dv"]).all(), "Nonfinite cached data")
    return matrix, anchor


def overlap_mask(ids, contexts, sigs, generic_hashes, generic_sigs, train_hashes, train_sigs):
    """Apply the predeclared operational exact and near-content exclusion."""
    require(set(ids) <= contexts.keys(), "Incomplete evaluation prompt-text join")
    components = {}
    for label, hashes in (("exact_generic", generic_hashes), ("exact_trait", train_hashes)):
        components[label] = np.array(
            [
                contexts[c]["question_normalized_sha256"] in hashes
                or bool(set(contexts[c]["user_turn_hashes"]) & hashes)
                for c in ids
            ]
        )
    es = np.asarray([sigs[c] for c in ids])
    components["near_generic"] = near_dup_mask(generic_sigs, es)
    components["near_trait"] = near_dup_mask(train_sigs, es)
    return ~np.logical_or.reduce(list(components.values())), components


def numeric_check(matrix, dv, groups, draws, seed):
    """Independently re-rank all point estimates and five sampled group draws."""
    estimates = arms.spearman_rows(matrix, dv)
    point = np.asarray([spearmanr(row, dv).statistic for row in matrix])
    np.testing.assert_allclose(estimates, point, atol=1e-12, rtol=0, equal_nan=True)
    ug = sorted(set(groups.tolist()))
    members = [np.flatnonzero(groups == group) for group in ug]
    choices = np.random.default_rng(seed).integers(0, len(ug), size=(500, len(ug)))
    maximum, checks = 0.0, 0
    for draw in (0, 1, 97, 249, 499):
        rows = np.concatenate([members[g] for g in choices[draw]])
        expected = np.asarray([spearmanr(row[rows], dv[rows]).statistic for row in matrix])
        np.testing.assert_allclose(draws[:, draw], expected, atol=1e-12, rtol=0, equal_nan=True)
        finite = np.isfinite(expected)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(draws[finite, draw] - expected[finite]))))
        checks += len(expected)
    return {
        "point_checks": len(point),
        "sampled_bootstrap_checks": checks,
        "max_abs_error": maximum,
    }


def estimate(value, draws, key="rho"):
    """Report finite percentile intervals without substituting absent estimates."""
    valid = np.isfinite(draws)
    return {
        key: float(value) if np.isfinite(value) else None,
        "ci95": np.quantile(draws[valid], [0.025, 0.975]).tolist() if valid.any() else None,
        "valid_bootstraps": int(valid.sum()),
    }


def differences(values, draws):
    """Keep E1-vs-natural paired differences for all four primary readouts."""
    pairs = [(f"q01_s0/{m}", f"e1/{m}") for m in METHODS] + [
        ("q01_s0/mapped_answer", f"q01_s0/{m}")
        for m in ("context_native", "answer_direction_on_context")
    ]
    return {
        f"{a}_minus_{b}": estimate(
            values[SERIES.index(a)] - values[SERIES.index(b)],
            draws[SERIES.index(a)] - draws[SERIES.index(b)],
            "delta",
        )
        for a, b in pairs
    }


def analyze_behavior(run, out, behavior, config, generic_hashes, generic_sigs, provenance):
    """Checkpoint every dataset, preserving original source-qualified bootstrap groups."""
    contexts, sigs = load_contexts(run, behavior, config, out, provenance)
    train = {k: c for k, c in contexts.items() if c["split"] == "train" and c["rung"] == "train"}
    require(len(train) == (6468 if behavior == "evil" else 16000), "Wrong trait map-fit row count")
    train_hashes = {c["question_normalized_sha256"] for c in train.values()} | {
        h for c in train.values() for h in c["user_turn_hashes"]
    }
    train_sigs = np.asarray([sigs[k] for k in train])
    source = run / "results" / behavior
    fetch = read_json(source / "fetch_manifest.json", provenance)
    original = read_json(
        source / "results.json", provenance, fetch["files"]["results.json"]["sha256"]
    )
    require(original["source_sha"] == fetch["source_sha"], "Science source mismatch")
    require(original["input_fingerprint"] == fetch["input_fingerprint"], "Science input mismatch")
    originals = {r["dataset"]: r for r in original["datasets"]}
    datasets = sorted(REGISTERED_PRIMARY_RUNGS[behavior] | {"heldin_train", "wildchat_rung"})
    require(set(datasets) == set(originals), "Original dataset roster mismatch")
    first_layer = min(LAYERS[behavior].values())
    selection_name = f"L{first_layer:02d}/selection.json"
    selection = read_json(
        source / selection_name, provenance, fetch["files"][selection_name]["sha256"]
    )
    exposure = {}
    for dataset in datasets:
        fold = "heldin:train" if dataset == "heldin_train" else dataset
        chosen = selection[f"{fold}/q01_s0"]
        require(chosen["status"] == "ok", f"Missing primary extraction: {fold}")
        exposure[dataset] = {}
        for tail in ("high", "low"):
            ids = chosen[f"{tail}_ids"]
            keep, components = overlap_mask(
                ids, contexts, sigs, generic_hashes, generic_sigs, train_hashes, train_sigs
            )
            exposure[dataset][tail] = {
                "n": len(ids),
                "direct_trait_map_ids": len(set(ids) & train.keys()),
                "any_map_content_exposure": int((~keep).sum()),
                "component_counts_nonexclusive": {k: int(v.sum()) for k, v in components.items()},
                "ids": ids,
                "operational_map_content_disjoint": keep.tolist(),
            }
    write_json(out / behavior / "extraction_map_exposure.json", exposure)
    summaries, bootstraps = [], {}
    for dataset in datasets:
        matrix, anchor = load_predictions(source, behavior, dataset, fetch, provenance)
        ids = anchor["ids"].tolist()
        full = arms.spearman_rows(matrix, anchor["dv"])
        np.testing.assert_allclose(
            full, [originals[dataset]["estimates"][s]["rho"] for s in SERIES], atol=1e-12, rtol=0
        )
        keep, components = overlap_mask(
            ids, contexts, sigs, generic_hashes, generic_sigs, train_hashes, train_sigs
        )
        target = out / behavior
        maskpath = target / "masks" / f"{dataset}.npz"
        atomic_npz(maskpath, ids=anchor["ids"], groups=anchor["groups"], keep=keep, **components)
        summary = {
            "dataset": dataset,
            "n_original": len(ids),
            "n": int(keep.sum()),
            "n_excluded": int((~keep).sum()),
            "joined_prompt_ids": len(ids),
            "n_groups": len(set(anchor["groups"][keep].tolist())),
            "mask_sha256": sha(maskpath),
            "exclusion_counts_nonexclusive": {k: int(v.sum()) for k, v in components.items()},
        }
        if dataset == "heldin_train":
            require(not keep.any(), "Unexpected map-disjoint ID rows")
            summary.update(
                status="unavailable",
                reason=RECIPE["id"],
                estimates={s: {"rho": None, "ci95": None} for s in SERIES},
                differences={},
            )
        else:
            require(
                keep.sum() >= 3 and len(np.unique(anchor["dv"][keep])) > 1,
                f"Degenerate retained dataset: {dataset}",
            )
            filtered, dv = matrix[:, keep], anchor["dv"][keep]
            # Dataset qualification prevents accidental collisions across sources.
            groups = np.asarray([f"{behavior}:{dataset}:{g}" for g in anchor["groups"][keep]])
            seed = int(hashlib.sha256(f"predictions_{dataset}.npz".encode()).hexdigest()[:16], 16)
            values = arms.spearman_rows(filtered, dv)
            draws, n_groups = group_bootstrap_rhos(
                filtered, dv, groups, n_boot=500, rng=np.random.default_rng(seed)
            )
            require(n_groups == summary["n_groups"], "Source qualification changed group count")
            check = numeric_check(filtered, dv, groups, draws, seed)
            atomic_npz(target / f"bootstrap_{dataset}.npz", methods=np.asarray(SERIES), draws=draws)
            summary.update(
                status="ok",
                bootstrap_seed=seed,
                numeric_check=check,
                estimates={s: estimate(values[i], draws[i]) for i, s in enumerate(SERIES)},
                differences=differences(values, draws),
            )
            bootstraps[dataset] = draws
        summaries.append(summary)
        write_json(target / "dataset_summary.json", summaries)
        write_json(out / "provenance.json", provenance)
        print(f"[overlap] {behavior}/{dataset} retained={summary['n']}/{len(ids)}", flush=True)
    ood = [r for r in summaries if r["dataset"] in REGISTERED_PRIMARY_RUNGS[behavior]]
    require(len(ood) == len(REGISTERED_PRIMARY_RUNGS[behavior]), "Incomplete OOD coverage")
    values = np.mean([[r["estimates"][s]["rho"] for s in SERIES] for r in ood], axis=0)
    draws = np.mean([bootstraps[r["dataset"]] for r in ood], axis=0)
    atomic_npz(out / behavior / "bootstrap_ood.npz", methods=np.asarray(SERIES), draws=draws)
    result = {
        "source_sha": original["source_sha"],
        "input_fingerprint": original["input_fingerprint"],
        "primary_revision": fetch["verified_revision"],
        "behavior": behavior,
        "primary": "q01_s0",
        "datasets": summaries,
        "ood": {s: estimate(values[i], draws[i]) for i, s in enumerate(SERIES)},
        "ood_differences": differences(values, draws),
        "uncertainty": RECIPE["conditioning"],
        "trait_map_rows": len(train),
        "trait_text_join_coverage": 1.0,
        "extraction_map_exposure": "extraction_map_exposure.json; extraction unchanged in this sensitivity",
    }
    write_json(out / behavior / "results.json", result)
    return result


def run_analysis(run, out):
    """Execute the authorized metadata/prediction-only sensitivity with provenance."""
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}
    script_sha = record(Path(__file__), provenance)
    config = read_json(ROOT / "configs/experiments/1739_natural_extremes.json", provenance)
    for path in (
        "scripts/issue1739_claim4_fold.py",
        "scripts/issue1739_natural_score.py",
        "src/explore_persona_space/experiments/issue_1739/arms.py",
        "src/explore_persona_space/experiments/issue_1739/corpus_staging.py",
        "src/explore_persona_space/experiments/issue_1739/store_io.py",
    ):
        record(ROOT / path, provenance)
    write_json(out / "recipe.json", RECIPE)
    hashes, sigs, generic = generic_pool(out, provenance)
    results = {}
    for behavior in BEHAVIORS:
        results[behavior] = analyze_behavior(run, out, behavior, config, hashes, sigs, provenance)
    write_json(out / "provenance.json", provenance)
    outputs = {
        p.relative_to(out).as_posix(): sha(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "complete.json"
    }
    completion = {
        "status": "complete",
        "analysis_script_sha256": script_sha,
        "science_source_sha": sorted({r["source_sha"] for r in results.values()}),
        "finished_at": time.time(),
        "elapsed_seconds": time.time() - started,
        "generic": generic,
        "ood_datasets_realized": sum(len(REGISTERED_PRIMARY_RUNGS[b]) for b in BEHAVIORS),
        "files": outputs,
    }
    write_json(out / "complete.json", completion)
    return completion


def main():
    """CLI for a complete same-question re-analysis of archived predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.run_root / "map_content_disjoint"
    run_analysis(args.run_root, out)


if __name__ == "__main__":
    main()
