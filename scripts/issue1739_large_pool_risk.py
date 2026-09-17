"""Freeze large-pool retrieval sets and stage cached answers; do not judge them."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(Path("/home/thomasjiralerspong/explore-persona-space/.env"))

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from explore_persona_space.orchestrate.hub import retry_transient
from scripts.issue1739_covariance_ablation import sha256, write_json
from scripts.issue1739_covariance_monitor import observe
from scripts.issue1739_million_monitor import run_phase, verify_source

REPO = "superkaiba1/explore-persona-space-data"
REVISION = "9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5"
PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m/raw_completions"
BEHAVIORS = ("evil", "sycophancy", "hallucination")
METHODS = {
    "preimage_cosine": (1, 0),
    "context_native_cosine": (1, 2),
    "answer_on_context_cosine": (1, 3),
    "mapped_answer_projection": (0, 1),
}


def normalized_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).strip().encode()).hexdigest()


def canonical_pool(rows, cis, hashes):
    eligible = np.flatnonzero(cis[rows] >= 0)
    eligible = eligible[np.argsort(rows[eligible], kind="stable")]
    _, first = np.unique(hashes[eligible], return_index=True)
    return eligible[np.sort(first)]


def ranked_top(scores, pool, rows, k):
    if not np.isfinite(scores[pool]).all():
        raise ValueError("Nonfinite ranking score")
    return pool[np.lexsort((rows[pool], -scores[pool]))[:k]]


def raw_path(ci):
    if not 0 <= ci < 960000:
        raise ValueError(f"Not a new-pool context: {ci}")
    return f"{PREFIX}/shard{ci // 30000:02d}_chunk{(ci % 30000) // 500:04d}.json"


def verify_blob(path, record):
    data = Path(path).read_bytes()
    digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if len(data) != record["size"] or digest != record["blob_id"]:
        raise ValueError(f"Pinned source blob mismatch: {record['path']}")
    return data


def progress(config, phase, **kwargs):
    out = Path(config["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    row = dict(phase=phase, checked_at=time.time(), source_sha=config["source_sha"], **kwargs)
    write_json(out / "progress.json", row)
    print(json.dumps(row), flush=True)


def finish(config, phase):
    directory = Path(config["outputs"]) / phase
    files = {
        p.name: sha256(p)
        for p in sorted(directory.iterdir())
        if p.is_file() and p.name != "complete.json"
    }
    write_json(
        directory / "complete.json",
        dict(
            source_sha=config["source_sha"],
            completed_at=time.time(),
            artifact_sha256=files,
            behavior_scoring_complete=False,
        ),
    )


def select(config):
    out = Path(config["outputs"]) / "selection"
    out.mkdir(parents=True, exist_ok=True)
    source = Path(config["retrieval"])
    done = json.loads((source / "complete.json").read_text())
    for name, digest in done["artifact_sha256"].items():
        if sha256(source / name) != digest:
            raise ValueError(f"Cached score mismatch: {name}")
    scores = np.load(source / "generic_scores.npy", mmap_mode="r")
    rows = np.load(source / "generic_score_rows.npy")
    bank = Path(config["map_inputs"])
    prepared = json.loads((bank / "prepare_complete.json").read_text())
    assert prepared["complete"] and prepared["regime"]["revision"] == REVISION
    for name in ("ci.npy", "split.npz", "map_prompt_hashes.npz"):
        if sha256(bank / name) != prepared["array_sha256"][name]:
            raise ValueError(f"Map input provenance changed: {name}")
    cis = np.load(bank / "ci.npy")
    hashes = np.load(bank / "map_prompt_hashes.npz")["normalized_sha256"]
    np.testing.assert_array_equal(rows, np.load(bank / "split.npz")["train"])
    assert scores.shape == (963444, 2, 12) and len(hashes) == len(rows)
    assert np.sum(cis[rows] >= 0) == 959844
    pool = canonical_pool(rows, cis, hashes)
    membership = []
    for bidx, behavior in enumerate(BEHAVIORS):
        for method, (metric, col) in METHODS.items():
            take = ranked_top(scores[:, metric, bidx * 4 + col], pool, rows, config["top_k"])
            for rank, i in enumerate(take, 1):
                membership.append(
                    dict(
                        behavior=behavior,
                        method=method,
                        rank=rank,
                        score=float(scores[i, metric, bidx * 4 + col]),
                        score_row=int(i),
                        map_row=int(rows[i]),
                        ci=int(cis[rows[i]]),
                    )
                )
    rng = np.random.default_rng(config["seed"])
    random = rng.choice(pool, size=config["n_random"], replace=False)
    for i in random:
        membership.append(
            dict(
                behavior="all",
                method="random",
                rank=None,
                score=None,
                score_row=int(i),
                map_row=int(rows[i]),
                ci=int(cis[rows[i]]),
            )
        )
    selected = sorted({r["score_row"] for r in membership})
    contexts = [
        dict(
            score_row=int(i),
            map_row=int(rows[i]),
            ci=int(cis[rows[i]]),
            normalized_sha256=hashes[i].decode(),
            raw_path=raw_path(int(cis[rows[i]])),
        )
        for i in selected
    ]
    summary = dict(
        n_map_rows=len(rows),
        n_response_available_rows=959844,
        n_unique_candidate_prompts=len(pool),
        n_top_per_method=config["top_k"],
        n_random=config["n_random"],
        seed=config["seed"],
        n_selected_unique_contexts=len(contexts),
        n_required_source_chunks=len({r["raw_path"] for r in contexts}),
        pool_is_map_training_data=True,
        answers_are_map_training_targets=True,
        behavior_scoring_complete=False,
        source_revision=REVISION,
        source_score_sha256=done["artifact_sha256"]["generic_scores.npy"],
        map_input_sha256={
            n: sha256(bank / n) for n in ("ci.npy", "split.npz", "map_prompt_hashes.npz")
        },
        methods=METHODS,
    )
    write_json(out / "summary.json", summary)
    write_json(out / "memberships.json", membership)
    write_json(out / "contexts.json", contexts)
    finish(config, "selection")
    progress(config, "selection_complete", **summary)


def stage(config):
    out = Path(config["outputs"]) / "responses"
    out.mkdir(parents=True, exist_ok=True)
    selected = json.loads((Path(config["outputs"]) / "selection/contexts.json").read_text())
    selection_dir = Path(config["outputs"]) / "selection"
    selection_done = json.loads((selection_dir / "complete.json").read_text())
    assert selection_done["source_sha"] == config["source_sha"]
    for name, digest in selection_done["artifact_sha256"].items():
        assert sha256(selection_dir / name) == digest, name
    inventory = json.loads(Path(config["inventory"]).read_text())
    assert inventory["revision"] == REVISION and inventory["repo_id"] == REPO
    records = {r["path"]: r for r in inventory["files"]}
    grouped = {}
    for row in selected:
        grouped.setdefault(row["raw_path"], {})[row["ci"]] = row
    byte_total = sum(records[path]["size"] for path in grouped)
    progress(config, "fetching_responses", n_chunks=len(grouped), source_bytes=byte_total)

    def fetch(path):
        local = retry_transient(
            lambda: hf_hub_download(
                REPO,
                path,
                repo_type="dataset",
                revision=REVISION,
                cache_dir=config["download_cache"],
            ),
            what=f"large-pool response {Path(path).name}",
        )
        raw = verify_blob(local, records[path])
        doc = json.loads(raw)
        wanted = grouped[path]
        found = {}
        for r in doc["rows"]:
            ci = int(r["ci"])
            if raw_path(ci) != path:
                raise ValueError(f"Unexpected ci/shard assignment: {path}/{ci}")
            if ci not in wanted:
                continue
            expected = wanted[ci]
            if ci in found or normalized_hash(r["prompt"]) != expected["normalized_sha256"]:
                raise ValueError(f"Bad prompt identity or duplicate: {path}/{ci}")
            if not isinstance(r["response"], str) or not r["response"].strip():
                raise ValueError(f"Expected a captured nonempty answer: {path}/{ci}")
            found[ci] = dict(
                **expected,
                prompt=r["prompt"],
                response=r["response"],
                response_sha256=hashlib.sha256(r["response"].encode()).hexdigest(),
            )
        if set(found) != set(wanted):
            raise ValueError(f"Missing selected context in {path}")
        return list(found.values()), dict(
            path=path,
            sha256=hashlib.sha256(raw).hexdigest(),
            blob_id=records[path]["blob_id"],
            size=len(raw),
        )

    responses, sources = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, path) for path in sorted(grouped)]
        for future in as_completed(futures):
            found, source = future.result()
            responses.extend(found)
            sources.append(source)
            if len(sources) % 20 == 0:
                progress(
                    config,
                    "fetching_responses",
                    completed_chunks=len(sources),
                    total_chunks=len(grouped),
                    n_responses=len(responses),
                )
    responses.sort(key=lambda r: r["ci"])
    assert len(responses) == len(selected)
    # Fixed order independent of methods/ranks; no rank or score in judge inputs.
    rng = np.random.default_rng(config["seed"] + 1)
    order = rng.permutation(len(responses))
    blind = [
        dict(
            item_id=f"pool-{responses[i]['ci']:06d}",
            prompt=responses[i]["prompt"],
            response=responses[i]["response"],
        )
        for i in order
    ]
    for prefix, items in [("selected_responses", responses), ("blind_judge_inputs", blind)]:
        part, chunks, count = 0, [], 0
        for item in items:
            line = json.dumps(item, ensure_ascii=False) + "\n"
            if count + len(line.encode()) > 8_000_000 and chunks:
                (out / f"{prefix}_{part:03d}.jsonl").write_text("".join(chunks))
                part, chunks, count = part + 1, [], 0
            chunks.append(line)
            count += len(line.encode())
        if chunks:
            (out / f"{prefix}_{part:03d}.jsonl").write_text("".join(chunks))
    write_json(out / "source_blobs.json", sorted(sources, key=lambda r: r["path"]))
    write_json(
        out / "summary.json",
        dict(
            n_contexts=len(responses),
            n_source_chunks=len(sources),
            cached_rollouts_per_context=1,
            source_bytes=byte_total,
            behavior_scoring_complete=False,
        ),
    )
    finish(config, "responses")
    progress(config, "responses_staged", n_contexts=len(responses), n_source_chunks=len(sources))


def archive(config):
    verify_source(config)
    out = Path(config["outputs"])
    for phase in ("selection", "responses"):
        done = json.loads((out / phase / "complete.json").read_text())
        assert done["source_sha"] == config["source_sha"]
        for name, digest in done["artifact_sha256"].items():
            assert sha256(out / phase / name) == digest
    for rel in config["source_files"]:
        target = out / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    shutil.copyfile(config["inventory"], out / "input_response_inventory.json")
    shutil.copyfile(Path(config["state_dir"]) / "config.json", out / "run_config.json")
    write_json(
        out / "preparation_complete.json",
        dict(
            source_sha=config["source_sha"],
            completed_at=time.time(),
            behavior_scoring_complete=False,
            pending="Behavior judging requires a working authorized non-Claude endpoint",
        ),
    )
    files = {str(p.relative_to(out)): p for p in out.rglob("*") if p.is_file()}
    api = HfApi()
    info = retry_transient(
        lambda: api.upload_folder(
            repo_id=REPO,
            repo_type="dataset",
            folder_path=out,
            path_in_repo=config["hf_prefix"],
            commit_message="Issue1739 frozen large-pool risk retrieval and responses",
        ),
        what="large-pool retrieval archive",
    )
    tree = retry_transient(
        lambda: list(
            api.list_repo_tree(
                REPO, config["hf_prefix"], repo_type="dataset", revision=info.oid, recursive=True
            )
        ),
        what="verify large-pool archive",
    )
    remote = {r.path: r for r in tree if hasattr(r, "size")}
    assert set(remote) == {config["hf_prefix"] + "/" + p for p in files}
    for name, path in files.items():
        row = remote[config["hf_prefix"] + "/" + name]
        assert row.size == path.stat().st_size
        if row.lfs:
            assert row.lfs.sha256 == sha256(path)
        else:
            verify_blob(path, dict(path=row.path, size=row.size, blob_id=row.blob_id))
    result = dict(
        source_sha=config["source_sha"],
        verified_revision=info.oid,
        preparation_complete=True,
        behavior_scoring_complete=False,
        n_files=len(files),
        prefix=config["hf_prefix"],
    )
    write_json(Path(config["state_dir"]) / "upload_verified.json", result)
    observe(config, "complete", {"status": "done"}, results=result)
    print(json.dumps(result), flush=True)


def main():
    # Config-only phase driver, consistent with the existing issue supervisor.
    config = json.loads(Path(sys.argv[1]).read_text())
    phase = sys.argv[2]
    verify_source(config)
    if phase == "monitor":
        try:
            for phase_config in config["phases"]:
                run_phase(config, phase_config)
            # run_phase emits a phase-finished observation; restore verified terminal state.
            result = json.loads((Path(config["state_dir"]) / "upload_verified.json").read_text())
            observe(config, "complete", {"status": "done"}, results=result)
        except BaseException as exc:
            observe(config, "backend_failed", {"status": "failed", "error": type(exc).__name__})
            raise
    else:
        {"select": select, "stage": stage, "archive": archive}[phase](config)


if __name__ == "__main__":
    main()
