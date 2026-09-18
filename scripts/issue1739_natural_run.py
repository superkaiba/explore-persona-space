"""Monitored remote CPU staging, fixed-layer scoring and verified preservation.

Usage: python scripts/issue1739_natural_run.py CONFIG_JSON BEHAVIOR [LAYER]
The layer argument is an internal isolated worker; the parent checkpoints and
uploads each finished layer before advancing. No model or judge calls occur.
"""

from __future__ import annotations

# ruff: noqa: E402 -- source-root bootstrap precedes project imports
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explore_persona_space.orchestrate import hub
from explore_persona_space.orchestrate.env import load_dotenv
from scripts.issue1739_natural_score import LAYERS, progress, score_layer, sha, write_json
from scripts.issue1739_natural_stage import stage


def identity(value):
    """Machine-stable identity of parameters and immutable input descriptors."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def stage_signatures(config, paths):
    """Checkpoint exact inherited MinHash signatures in bounded 128-text chunks."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739.corpus_staging import minhash_signatures

    metadata = json.loads(Path(paths["prompt_metadata"]).read_text())
    if "context_parts" in metadata:
        if "contexts" in metadata:
            raise ValueError("Ambiguous inline and sharded contexts")
        contexts = {}
        for part in metadata["context_parts"]:
            path = Path(paths["prompt_metadata"]).parent / part["path"]
            if sha(path) != part["sha256"]:
                raise ValueError(f"Context shard hash mismatch: {path}")
            rows = json.loads(path.read_text())
            if len(rows) != part["n"] or set(rows).intersection(contexts):
                raise ValueError(f"Context shard identity/count mismatch: {path}")
            contexts.update(rows)
        if len(contexts) != metadata["n_contexts"]:
            raise ValueError("Context shard union count mismatch")
        metadata["contexts"] = contexts
    keys = sorted(metadata["contexts"])
    target = Path(config["out_root"]) / "metadata"
    target.mkdir(parents=True, exist_ok=True)
    fingerprint = identity(
        {
            "metadata_sha": sha(paths["prompt_metadata"]),
            "recipe": "corpus_staging.minhash_signatures",
            "n_perm": 64,
            "seed": 0,
            "source_sha": config["source_sha"],
        }
    )
    arrays = []
    receipt = target / "complete.json"
    verified = json.loads(receipt.read_text()) if receipt.exists() else None
    expected_names = {f"signatures_{i:06d}.npz" for i in range(0, len(keys), 128)}
    for name in expected_names:
        # This attempt owns these uncommitted chunk siblings. A crash before
        # atomic rename must not turn a resumable write into a permanent gate.
        (target / name).with_suffix(".partial.npz").unlink(missing_ok=True)
    existing_names = {p.name for p in target.glob("signatures_*.npz")}
    if existing_names - expected_names:
        raise ValueError("Unexpected signature chunks")
    if verified is not None and (
        verified["fingerprint"] != fingerprint or set(verified["files"]) != expected_names
    ):
        raise ValueError("Metadata completion manifest mismatch")
    for start in range(0, len(keys), 128):
        part = target / f"signatures_{start:06d}.npz"
        ck = keys[start : start + 128]
        if part.exists():
            if verified is not None and sha(part) != verified["files"][part.name]:
                raise ValueError(f"Signature chunk hash mismatch: {part}")
            with np.load(part) as archive:
                if str(archive["fingerprint"]) != fingerprint or archive["ids"].tolist() != ck:
                    raise ValueError(f"Stale metadata chunk: {part}")
                array = archive["signatures"]
                if array.shape != (len(ck), 64) or array.dtype != np.uint64:
                    raise ValueError(f"Invalid signature chunk shape/dtype: {part}")
        else:
            array = minhash_signatures([metadata["contexts"][c]["query"] for c in ck], seed=0)
            temp = part.with_suffix(".partial.npz")
            np.savez(temp, fingerprint=fingerprint, ids=np.asarray(ck), signatures=array)
            temp.replace(part)
        arrays.append(array)
        progress(
            config, "metadata_signatures", contexts_done=start + len(ck), contexts_total=len(keys)
        )
    metadata["signatures"] = dict(zip(keys, np.concatenate(arrays).tolist(), strict=True))
    write_json(
        target / "complete.json",
        {
            "source_sha": config["source_sha"],
            "fingerprint": fingerprint,
            "n": len(keys),
            "files": {p.name: sha(p) for p in target.glob("signatures_*.npz")},
        },
    )
    return metadata


def reference_rows(path):
    """Read exact original per-dataset values rather than rounded figure bars."""
    result = {}
    for cell in json.loads(Path(path).read_text())["cells"]:
        for arm, summary in cell["arms"].items():
            for dataset, row in summary["dataset_rows"].items():
                key = (dataset, arm)
                if key in result and result[key] != row:
                    raise ValueError(f"Conflicting frozen reference row: {key}")
                result[key] = row
    return result


def upload_verified(config, phase):
    """Upload all current outputs; independently verify exact remote name/hash set."""
    from huggingface_hub import HfApi

    root = Path(config["out_root"])
    api = HfApi()
    prefix = config["upload_prefix"]
    files = {
        p.relative_to(root).as_posix(): p
        for p in root.rglob("*")
        if p.is_file() and ".partial" not in p.name and p.name != "upload_verified.json"
    }
    digests = {name: sha(path) for name, path in files.items()}
    commit = hub.retry_transient(
        lambda: api.upload_folder(
            repo_id=hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
            folder_path=str(root),
            path_in_repo=prefix,
            ignore_patterns=["*.partial*", "upload_verified.json"],
            commit_message=f"#1739 natural extremes {config['behavior']} {phase}",
        ),
        what=f"natural-extremes upload {phase}",
    )
    revision = commit.oid
    remote = hub.retry_transient(
        lambda: list(
            api.list_repo_tree(
                hub.DEFAULT_DATASET_REPO,
                path_in_repo=prefix,
                repo_type="dataset",
                revision=revision,
                recursive=True,
            )
        ),
        what="natural-extremes independent upload verification",
    )
    entries = {r.path[len(prefix) + 1 :]: r for r in remote if hasattr(r, "size")}
    if set(entries) != set(files):
        raise ValueError(f"Remote output name-set mismatch at {revision}")
    for name, path in files.items():
        entry = entries[name]
        if entry.size != path.stat().st_size:
            raise ValueError(f"Remote output size mismatch: {name}")
        if entry.lfs is not None:
            if entry.lfs.sha256 != digests[name]:
                raise ValueError(f"Remote LFS SHA mismatch: {name}")
        else:
            digest = hashlib.sha1(f"blob {entry.size}\0".encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != entry.blob_id:
                raise ValueError(f"Remote Git blob mismatch: {name}")
    report = {
        "source_sha": config["source_sha"],
        "verified_revision": revision,
        "prefix": prefix,
        "time": time.time(),
        "n_files": len(files),
        "sha256": digests,
        "phase": phase,
    }
    write_json(root / "upload_verified.json", report)
    return report


def publish_completion(config, verification):
    """Publish independently readable completion after the exact output-set gate."""
    from huggingface_hub import HfApi, hf_hub_download

    receipt = {
        "source_sha": config["source_sha"],
        "behavior": config["behavior"],
        "input_fingerprint": config["input_fingerprint"],
        "status": "complete",
        "time": time.time(),
        "verified_revision": verification["verified_revision"],
        "verification": verification,
    }
    path = config["upload_prefix"] + ".completion.json"
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    commit = hub.retry_transient(
        lambda: HfApi().upload_file(
            path_or_fileobj=payload,
            path_in_repo=path,
            repo_id=hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"#1739 verified completion {config['behavior']}",
        ),
        what="natural-extremes completion receipt",
    )
    downloaded = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
            filename=path,
            revision=commit.oid,
        ),
        what="natural-extremes completion receipt readback",
    )
    if Path(downloaded).read_bytes() != payload:
        raise ValueError("Completion receipt readback differs from published bytes")
    receipt["receipt_revision"] = commit.oid
    return receipt


def summarize(config):  # noqa: C901 -- explicit coverage and missing-statistic handling
    """Pair rows across frozen layers and bootstrap source groups per dataset."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739.arms import spearman_rows
    from scripts.issue1739_claim4_fold import group_bootstrap_rhos

    out = Path(config["out_root"])
    files = {}
    required_layers = set(LAYERS[config["behavior"]].values())
    reference = reference_rows(Path(config["metadata_root"]) / "reference.json")
    required_files = {f"predictions_{ds.replace(':', '_')}.npz" for ds, _ in reference}
    for layer in sorted(required_layers):
        actual_files = {p.name for p in (out / f"L{layer:02d}").glob("predictions_*.npz")}
        if actual_files != required_files:
            raise ValueError(f"Missing/unexpected evaluation datasets at layer {layer}")
        for p in (out / f"L{layer:02d}").glob("predictions_*.npz"):
            files.setdefault(p.name, {})[layer] = p
    summary, bootstrap_records = [], {}
    for filename, layer_paths in sorted(files.items()):
        if set(layer_paths) != required_layers:
            raise ValueError(f"Incomplete frozen-layer coverage: {filename}")
        predictions, methods, anchor = [], [], None
        for layer, path in layer_paths.items():
            with np.load(path) as data:
                content = {k: data[k] for k in data.files}
            if anchor is None:
                anchor = content
            elif any(not np.array_equal(anchor[k], content[k]) for k in ("ids", "groups", "dv")):
                raise ValueError(f"Frozen-layer evaluation row mismatch: {filename}")
            for i, (variant, arm) in enumerate(
                zip(content["variants"], content["arms"], strict=True)
            ):
                if layer == LAYERS[config["behavior"]][arm] and not variant.startswith("half"):
                    predictions.append(content["predictions"][i])
                    methods.append(f"{variant}/{arm}")
        if len(methods) != len(set(methods)):
            raise ValueError(f"Duplicate summary method: {filename}")
        matrix = np.asarray(predictions)
        estimate = spearman_rows(matrix, anchor["dv"])
        rng = np.random.default_rng(int(hashlib.sha256(filename.encode()).hexdigest()[:16], 16))
        draws, n_groups = group_bootstrap_rhos(
            matrix, anchor["dv"], anchor["groups"], n_boot=500, rng=rng
        )
        dataset = filename.removeprefix("predictions_").removesuffix(".npz")
        estimates = {}
        for i, method in enumerate(methods):
            valid = np.isfinite(draws[i])
            estimates[method] = {
                "rho": float(estimate[i]) if np.isfinite(estimate[i]) else None,
                "ci95": np.quantile(draws[i, valid], [0.025, 0.975]).tolist()
                if valid.any()
                else None,
                "valid_bootstraps": int(valid.sum()),
            }
        differences = {}
        primary = "q01_s0/mapped_answer"
        for other in (
            "e1/mapped_answer",
            "q01_s0/context_native",
            "q01_s0/answer_direction_on_context",
        ):
            if primary not in methods or other not in methods:
                differences[f"{primary}_minus_{other}"] = {"status": "unavailable"}
                continue
            a, b = methods.index(primary), methods.index(other)
            delta = draws[a] - draws[b]
            valid = np.isfinite(delta)
            differences[f"{primary}_minus_{other}"] = {
                "delta": float(estimate[a] - estimate[b])
                if np.isfinite(estimate[[a, b]]).all()
                else None,
                "ci95": np.quantile(delta[valid], [0.025, 0.975]).tolist() if valid.any() else None,
                "valid_bootstraps": int(valid.sum()),
            }
        summary.append(
            {
                "dataset": dataset,
                "n": len(anchor["ids"]),
                "n_groups": n_groups,
                "estimates": estimates,
                "differences": differences,
            }
        )
        bootstrap_records[dataset] = (methods, draws)
        np.savez(out / f"bootstrap_{dataset}.npz", methods=np.asarray(methods), draws=draws)
        write_json(out / "dataset_summary.json", summary)
        progress(config, "bootstrap_dataset", dataset=dataset, n=len(anchor["ids"]))
    ood = [r for r in summary if r["dataset"] not in {"heldin_train", "wildchat_rung"}]
    all_methods = sorted({m for r in ood for m in r["estimates"]})
    mean_rows = {}
    for method in all_methods:
        present = [
            r for r in ood if method in r["estimates"] and r["estimates"][method]["rho"] is not None
        ]
        values = [r["estimates"][method]["rho"] for r in present]
        if len(present) != len(ood):
            mean_rows[method] = {
                "rho": None,
                "datasets_available": len(present),
                "datasets_planned": len(ood),
                "status": "incomplete",
            }
            continue
        means = np.mean(
            [
                bootstrap_records[r["dataset"]][1][bootstrap_records[r["dataset"]][0].index(method)]
                for r in present
            ],
            axis=0,
        )
        valid = np.isfinite(means)
        mean_rows[method] = {
            "rho": float(np.mean(values)),
            "dataset_sem": float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else None,
            "ci95": np.quantile(means[valid], [0.025, 0.975]).tolist() if valid.any() else None,
            "n_datasets": len(values),
            "valid_bootstraps": int(valid.sum()),
        }
    ood_differences = {}
    primary = "q01_s0/mapped_answer"
    for other in (
        "e1/mapped_answer",
        "q01_s0/context_native",
        "q01_s0/answer_direction_on_context",
    ):
        key = f"{primary}_minus_{other}"
        present = [
            r
            for r in ood
            if primary in r["estimates"]
            and other in r["estimates"]
            and r["estimates"][primary]["rho"] is not None
            and r["estimates"][other]["rho"] is not None
        ]
        coverage = {"datasets_available": len(present), "datasets_planned": len(ood)}
        if len(present) != len(ood):
            ood_differences[key] = {"delta": None, "ci95": None, "coverage": coverage}
            continue
        paired = []
        for row in present:
            methods, draws = bootstrap_records[row["dataset"]]
            paired.append(draws[methods.index(primary)] - draws[methods.index(other)])
        delta_draws = np.mean(paired, axis=0)
        valid = np.isfinite(delta_draws)
        ood_differences[key] = {
            "delta": float(
                np.mean(
                    [r["estimates"][primary]["rho"] - r["estimates"][other]["rho"] for r in present]
                )
            ),
            "ci95": np.quantile(delta_draws[valid], [0.025, 0.975]).tolist()
            if valid.any()
            else None,
            "valid_bootstraps": int(valid.sum()),
            "coverage": coverage,
        }
    result = {
        "source_sha": config["source_sha"],
        "input_fingerprint": config["input_fingerprint"],
        "behavior": config["behavior"],
        "datasets": summary,
        "ood": mean_rows,
        "ood_differences": ood_differences,
        "uncertainty": "500 paired group bootstraps; extraction and maps held fixed",
        "historical_answer_pooling": "all five cached answers; extraction uses judged-valid only",
        "primary": "q01_s0",
        "finished_at": time.time(),
    }
    write_json(out / "results.json", result)
    return result


def main():
    """Run one source-pinned behavior; refuse stale source or checkpoint identity."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    config_path, behavior = Path(sys.argv[1]).resolve(), sys.argv[2]
    base = json.loads(config_path.read_text())
    config = {**base["shared"], **base["behaviors"][behavior], "behavior": behavior}
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if config.get("source_sha") not in (None, source):
        raise ValueError(f"Unexpected source checkout {source}")
    config["source_sha"] = source
    config["input_fingerprint"] = identity(
        {k: v for k, v in config.items() if k not in {"store_root", "out_root", "metadata_root"}}
    )
    if os.environ.get("EPS_I1739_RIDGE_LAMBDAS"):
        raise ValueError("Original Figure6 requires the original ridge grid")
    out = Path(config["out_root"])
    if len(sys.argv) == 3:
        write_json(
            out / "worker.pid",
            {
                "source_sha": source,
                "pid": os.getpid(),
                "start_ticks": int(
                    Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
                ),
            },
        )
        write_json(out / "run_config.json", config)
        paths = stage(config)
        paths = {
            k: (
                [str(x) for x in v]
                if isinstance(v, list)
                else {a: str(b) for a, b in v.items()}
                if isinstance(v, dict)
                else str(v)
            )
            for k, v in paths.items()
        }
        write_json(out / "staged_paths.json", paths)
        stage_signatures(config, paths)
        for layer in sorted(set(LAYERS[behavior].values())):
            completed = subprocess.run(
                [sys.executable, __file__, str(config_path), behavior, str(layer)],
                cwd=ROOT,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(f"Layer {layer} worker exited {completed.returncode}")
            upload_verified(config, f"L{layer:02d}")
        summarize(config)
        write_json(
            out / "completion.json",
            {
                "source_sha": source,
                "status": "computed",
                "layers": sorted(set(LAYERS[behavior].values())),
                "time": time.time(),
                "input_fingerprint": config["input_fingerprint"],
            },
        )
        verification = upload_verified(config, "complete")
        published = publish_completion(config, verification)
        write_json(
            out / "run_complete.json",
            published,
        )
        progress(config, "complete", verified_revision=verification["verified_revision"])
    else:
        paths = json.loads((out / "staged_paths.json").read_text())
        paths["train_labels"] = paths["labels"]["train"]
        paths["wildchat_labels"] = paths["labels"]["wildchat"]
        if "ood" in paths["labels"]:
            paths["ood_labels"] = paths["labels"]["ood"]
        metadata = stage_signatures(config, paths)
        reference = reference_rows(Path(config["metadata_root"]) / "reference.json")
        record = score_layer(config, paths, int(sys.argv[3]), metadata, reference)
        # The real production-shape first layer is the pilot. Never weaken rows
        # or dimensions for the measured resource check.
        if record["max_rss_gib"] > 64 or record["wall_s"] > 7200:
            raise RuntimeError(f"Production-layer pilot exceeds declared resources: {record}")


if __name__ == "__main__":
    main()
