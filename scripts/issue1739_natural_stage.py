"""Pinned, behavior-scoped input staging for the natural-extremes CPU run.

``stage(config)`` accepts behavior, store_root, out_root, metadata_root and
source_sha. It returns Path-valued labeling/wildchat/u_store/extraction roots,
an OOD-root list, labels, and optional prompt_metadata. All downloads
run off the shared VM. Importing this module performs no staging or API calls.
Sources can be overridden explicitly through label_sources;
each source is {kind: git|hf, path, revision, sha256?, repo?}. Overrides are
recorded, never discovered through a silent missing-file fallback. ``hf_files``
maps metadata-root-relative names (e.g. labels/train.json, prompts.jsonl) to
{repo, path, revision, sha256} descriptors and takes precedence over label defaults.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)
CAPTURE_REVISION = "7a47ff5ce42f16308bebaba29c1286a4e9bc8008"
DATA_REPO = "superkaiba1/explore-persona-space-data"
LAYERS = {"sycophancy": (11, 16), "hallucination": (23, 24, 27), "evil": (15, 22)}
KINDS = ("context_end", "t1")
OOD_PREFIXES = {
    "sycophancy": ("syco_ood/store",),
    "hallucination": (),
    "evil": tuple(
        f"evil_ood_full/store/{x}" for x in ("mhj", "pair", "tomgibbs_p0", "tomgibbs_p1")
    ),
}
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class _Progress:
    def __init__(self, path: Path, source_sha: str, behavior: str):
        self.path = path
        self.lock = threading.Lock()
        self.state = {"source_sha": source_sha, "behavior": behavior, "completed_files": 0}

    def emit(self, phase: str, *, completed: bool = False, **fields: Any) -> None:
        with self.lock:
            if completed:
                self.state["completed_files"] += 1
            self.state.update(phase=phase, checked_at=time.time(), **fields)
            _atomic_json(self.path, self.state)
            logger.info("[stage] %s", json.dumps(self.state, sort_keys=True))


@contextmanager
def _download_observer(root: Path, progress: _Progress, phase: str):
    """Expose real temporary/downloaded file bytes while HF owns its downloader."""
    stop = threading.Event()

    def observe() -> None:
        while not stop.wait(10):
            sizes = []
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    sizes.append(path.stat().st_size)
                except FileNotFoundError:
                    # Atomic download rename can race this read-only observer.
                    continue
            progress.emit(phase, observed_files=len(sizes), observed_bytes=sum(sizes))

    worker = threading.Thread(target=observe, name="natural-stage-observer", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=15)
        if worker.is_alive():
            raise RuntimeError("staging progress observer failed to stop")


def _hashes(path: Path, progress: _Progress | None = None) -> tuple[str, str]:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    copied = 0
    last = time.monotonic()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            sha256.update(chunk)
            git_blob.update(chunk)
            copied += len(chunk)
            if progress is not None and time.monotonic() - last >= 10:
                progress.emit("hash-input", file=str(path), hashed_bytes=copied, total_bytes=size)
                last = time.monotonic()
    return sha256.hexdigest(), git_blob.hexdigest()


def _check_space(root: Path, expected_bytes: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    required = int(1.5 * expected_bytes)
    if free < required:
        raise RuntimeError(f"insufficient staging space at {root}: free={free}, need={required}")


def _assert_remote() -> None:
    if socket.gethostname().startswith("cia-benchmark-vm") or Path("/mnt/eps-data").is_mount():
        raise RuntimeError("natural-extremes staging runs on a remote cpu-bigmem backend only")


def _wanted(name: str, kinds: tuple[str, ...], layers: tuple[int, ...]) -> bool:
    if re.fullmatch(r"row_index(?:_shard\d+)?\.jsonl", name):
        return True
    match = re.fullmatch(r"(context_end|t1)_L(\d{2})(?:_shard\d+)?\.npy", name)
    return match is not None and match[1] in kinds and int(match[2]) in layers


def _remote_rows(api, paths: list[str], *, revision: str, repo: str = DATA_REPO) -> dict:
    from explore_persona_space.orchestrate import hub

    result = {}
    for start in range(0, len(paths), 100):
        batch = paths[start : start + 100]
        rows = hub.retry_transient(
            lambda batch=batch: api.get_paths_info(
                repo, batch, repo_type="dataset", revision=revision
            ),
            what=f"natural-stage metadata {repo}@{revision}",
        )
        result.update({row.path: row for row in rows})
    if set(result) != set(paths):
        raise FileNotFoundError(f"pinned HF paths missing: {sorted(set(paths) - set(result))}")
    return result


def _verify_remote(path: Path, info, revision: str, progress: _Progress) -> dict:
    if path.stat().st_size != info.size:
        raise ValueError(f"size mismatch for {path}: {path.stat().st_size} != {info.size}")
    sha256, blob_sha1 = _hashes(path, progress)
    if info.lfs is not None:
        expected = info.lfs.sha256
        if sha256 != expected:
            raise ValueError(f"HF LFS SHA256 mismatch: {path}")
        algorithm = "lfs-sha256"
    else:
        expected = info.blob_id
        if blob_sha1 != expected:
            raise ValueError(f"HF git blob SHA1 mismatch: {path}")
        algorithm = "git-blob-sha1"
    record = {
        "local_path": str(path),
        "path": info.path,
        "revision": revision,
        "bytes": info.size,
        "sha256": sha256,
        "remote_hash": expected,
        "remote_hash_algorithm": algorithm,
    }
    progress.emit("verified-file", completed=True, file=str(path), bytes=info.size)
    return record


def _stage_hf(api, source: dict, target: Path, progress: _Progress, info=None) -> dict:
    from explore_persona_space.orchestrate import hub

    repo = source.get("repo", DATA_REPO)
    revision = source["revision"]
    if re.fullmatch(r"[0-9a-f]{7,40}", revision) is None:
        raise ValueError(f"HF source must use an immutable revision: {revision!r}")
    if info is None:
        info = _remote_rows(api, [source["path"]], revision=revision, repo=repo)[source["path"]]
    progress.emit("download-file", file=source["path"], total_bytes=info.size)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _download_observer(target.parent, progress, "download-file"):
        hub.stage_hub_file(
            repo,
            source["path"],
            target,
            repo_type="dataset",
            revision=revision,
            size_bytes=info.size,
        )
    record = {"kind": "hf", "repo": repo, **_verify_remote(target, info, revision, progress)}
    if source.get("sha256") and record["sha256"] != source["sha256"]:
        raise ValueError(f"configured SHA256 mismatch: {target}")
    return record


def _stage_source(api, source: dict, target: Path, progress: _Progress) -> dict:
    if source["kind"] == "hf":
        return _stage_hf(api, source, target, progress)
    if source["kind"] != "git":
        raise ValueError(f"unknown source kind: {source['kind']}")
    revision = source["revision"]
    if re.fullmatch(r"[0-9a-f]{7,40}", revision) is None:
        raise ValueError(f"git source must use a commit pin: {revision!r}")
    blob = subprocess.check_output(["git", "show", f"{revision}:{source['path']}"], cwd=_REPO_ROOT)
    sha256 = hashlib.sha256(blob).hexdigest()
    if source.get("sha256") and sha256 != source["sha256"]:
        raise ValueError(f"configured git-input SHA256 mismatch: {source['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, target)
    progress.emit("verified-git-file", completed=True, file=str(target), bytes=len(blob))
    return {
        "kind": "git",
        "revision": revision,
        "path": source["path"],
        "local_path": str(target),
        "sha256": sha256,
        "bytes": len(blob),
    }


def _extract_slice(
    archive: Path,
    dest: Path,
    kinds: tuple[str, ...],
    layers: tuple[int, ...],
    progress: _Progress,
) -> list[dict]:
    """Flatten selected regular members, rejecting collisions and unsafe paths."""
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    with tarfile.open(archive, mode="r|") as tar:
        for member in tar:
            logical = PurePosixPath(member.name)
            if logical.is_absolute() or ".." in logical.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            if not _wanted(logical.name, kinds, layers):
                continue
            if not member.isfile() or logical.name in seen:
                raise ValueError(f"non-file or colliding selected member: {member.name}")
            seen.add(logical.name)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable selected member: {member.name}")
            target = dest / logical.name
            tmp = target.with_name(target.name + ".tmp")
            digest = hashlib.sha256()
            copied = 0
            with stream, tmp.open("wb") as output:
                while chunk := stream.read(8 << 20):
                    output.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if copied != member.size:
                raise ValueError(f"truncated archive member: {member.name}")
            os.replace(tmp, target)
            records.append(
                {
                    "local_path": str(target),
                    "archive_member": member.name,
                    "bytes": copied,
                    "sha256": digest.hexdigest(),
                }
            )
            progress.emit("extract-member", completed=True, file=str(target), bytes=copied)
    if not records:
        raise ValueError(f"no requested members in {archive}")
    return records


def _validate_store(dest: Path, kinds: tuple[str, ...], layers: tuple[int, ...]) -> dict:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import store_io

    rows = store_io._index_rows_for(dest, list(kinds))
    if not rows:
        raise ValueError(f"empty row index: {dest}")
    counts = {}
    for kind in kinds:
        for layer in layers:
            _, paths = store_io._resolve_summary_kind(dest, kind, layer)
            count = 0
            for path in paths:
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                if values.ndim != 2 or values.shape[1] != 3584:
                    raise ValueError(f"invalid activation shape {path}: {values.shape}")
                if not np.issubdtype(values.dtype, np.floating):
                    raise ValueError(f"nonfloating activation dtype {path}: {values.dtype}")
                count += len(values)
                for start in range(0, len(values), 1024):
                    if not np.isfinite(values[start : start + 1024]).all():
                        raise ValueError(f"nonfinite activations: {path}")
            if count != len(rows):
                raise ValueError(
                    f"index/activation count mismatch {dest}/{kind}/L{layer}: "
                    f"{count} != {len(rows)}"
                )
            counts[f"{kind}_L{layer:02d}"] = count
    return {"root": str(dest), "rows": len(rows), "counts": counts}


def _stage_loose(
    api, prefix: str, dest: Path, layers: tuple[int, ...], progress: _Progress
) -> list[dict]:
    from explore_persona_space.orchestrate import hub

    entries = hub.retry_transient(
        lambda: list(
            api.list_repo_tree(
                DATA_REPO,
                path_in_repo=prefix,
                recursive=True,
                repo_type="dataset",
                revision=CAPTURE_REVISION,
            )
        ),
        what=f"natural-stage listing {prefix}",
    )
    paths = [
        entry.path for entry in entries if _wanted(PurePosixPath(entry.path).name, KINDS, layers)
    ]
    basenames = [PurePosixPath(path).name for path in paths]
    if not paths or len(set(basenames)) != len(paths):
        raise ValueError(f"empty or colliding loose-store selection: {prefix}")
    by_path = {entry.path: entry for entry in entries}

    def one(path: str) -> dict:
        return _stage_hf(
            api,
            {
                "path": path,
                "revision": CAPTURE_REVISION,
            },
            dest / PurePosixPath(path).name,
            progress,
            info=by_path[path],
        )

    with ThreadPoolExecutor(max_workers=6) as workers:
        return list(workers.map(one, sorted(paths)))


def _stage_metadata(
    api, config: dict, metadata_root: Path, progress: _Progress, records: list
) -> dict:
    behavior = config["behavior"]
    source_sha = config["source_sha"]
    paths = {"labels": {}}
    staged_extra = {}
    for relative, source in config.get("hf_files", {}).items():
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe metadata destination: {relative}")
        if not source.get("sha256"):
            raise ValueError(f"hf_files descriptor requires sha256: {relative}")
        target = metadata_root / relative_path
        records.append(_stage_hf(api, source, target, progress))
        staged_extra[relative] = target
    if "prompts.jsonl" in staged_extra:
        paths["prompt_metadata"] = staged_extra["prompts.jsonl"]
    default_sources = {
        "train": {
            "kind": "git",
            "revision": source_sha,
            "path": f"eval_results/issue_1739/dv_dataset/{behavior}/labeling.json",
        },
        "wildchat": {
            "kind": "hf",
            "revision": CAPTURE_REVISION,
            "path": f"issue1739_ctxmap/wildchat_rung/dv_dataset/{behavior}/labeling.json",
        },
    }
    if behavior != "hallucination":
        prefix = "evil_ood_full" if behavior == "evil" else "syco_ood"
        default_sources["ood"] = {
            "kind": "git",
            "revision": source_sha,
            "path": f"eval_results/issue_1739/{prefix}/dv_dataset/{behavior}/labeling.json",
        }
    overrides = config.get("label_sources", {})
    if set(overrides) - set(default_sources):
        raise ValueError(f"unknown label source roles: {set(overrides) - set(default_sources)}")
    for role, source in (default_sources | overrides).items():
        target = metadata_root / "labels" / f"{role}.json"
        if f"labels/{role}.json" not in staged_extra:
            records.append(_stage_source(api, source, target, progress))
        labels = json.loads(target.read_text())
        if labels.get("behavior") != behavior or not labels.get("rows"):
            raise ValueError(f"invalid behavior/empty label rows: {target}")
        paths["labels"][role] = target
    if behavior == "hallucination":
        target = metadata_root / "hallu_per_rollout.json"
        if "hallu_per_rollout.json" not in staged_extra:
            records.append(
                _stage_hf(
                    api,
                    {
                        "path": "issue1739_ctxmap/judge/hallucination/labeling_per_rollout.json",
                        "revision": CAPTURE_REVISION,
                    },
                    target,
                    progress,
                )
            )
        paths["hallu_per_rollout"] = target
    metadata_keys = {"metadata_repo", "metadata_path", "metadata_revision", "metadata_hash"}
    if metadata_keys & set(config):
        if not metadata_keys <= set(config):
            raise ValueError(
                f"partial prompt metadata configuration: missing {metadata_keys - set(config)}"
            )
        target = metadata_root / "prompts.jsonl"
        records.append(
            _stage_hf(
                api,
                {
                    "repo": config["metadata_repo"],
                    "path": config["metadata_path"],
                    "revision": config["metadata_revision"],
                    "sha256": config["metadata_hash"],
                },
                target,
                progress,
            )
        )
        paths["prompt_metadata"] = target
    return paths


def stage(config: dict) -> dict:
    """Stage and hash-verify one behavior; no generation, judging, or model load."""
    _assert_remote()
    behavior = config["behavior"]
    layers = LAYERS[behavior]
    source_sha = config["source_sha"]
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source_sha must be a full committed source SHA")
    roots = {
        key: Path(config[key]).resolve() for key in ("store_root", "out_root", "metadata_root")
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    store_root, metadata_root = roots["store_root"], roots["metadata_root"]
    progress = _Progress(roots["out_root"] / "staging_progress.json", source_sha, behavior)
    progress.emit("staging-start", status="running")
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from huggingface_hub import HfApi

    from explore_persona_space.experiments.issue_1739 import constants, store_io

    api = HfApi()
    records: list[dict] = []
    validations = []
    paths = {
        "labeling": store_root / f"{behavior}_labeling",
        "wildchat": store_root / "wildchat",
        "u_store": store_root / "u_store",
        "extraction": store_root / f"{behavior}_extraction",
        "ood": [store_root / relative for relative in OOD_PREFIXES[behavior]],
        "labels": {},
    }
    for suffix, kinds in (("labeling", KINDS), ("extraction", KINDS)):
        stem = f"{behavior}_{suffix}"
        hf_path = f"issue1739_ctxmap/capture_store/{stem}/{stem}.tar"
        archive = store_root / "archives" / f"{stem}.tar"
        info = _remote_rows(api, [hf_path], revision=CAPTURE_REVISION)[hf_path]
        missing_bytes = 0 if archive.exists() else info.size
        _check_space(
            store_root, missing_bytes + int(config.get("retained_reserve_bytes", 12 << 30))
        )
        records.append(
            _stage_hf(api, {"path": hf_path, "revision": CAPTURE_REVISION}, archive, progress)
        )
        extracted = _extract_slice(archive, paths[suffix], kinds, layers, progress)
        for record in extracted:
            record.update(archive=str(archive), revision=CAPTURE_REVISION)
        records.extend(extracted)
        validations.append(_validate_store(paths[suffix], kinds, layers))
        _atomic_json(
            paths[suffix] / "natural_slice_manifest.json",
            {
                "source_sha": source_sha,
                "revision": CAPTURE_REVISION,
                "layers": layers,
                "kinds": kinds,
                "files": extracted,
                "checked_at": time.time(),
            },
        )

    progress.emit("stage-u-store")
    with _download_observer(store_root, progress, "stage-u-store"):
        store_io.stage_u_store(
            paths["u_store"],
            kinds=KINDS,
            layers=layers,
            revision=constants.STORE_REVISION,
            manifest_path=constants.CORPUS_MANIFEST_PATH,
            manifest_revision=constants.CORPUS_MANIFEST_REVISION,
            max_workers=6,
        )
    u_manifest = json.loads((paths["u_store"] / "staging_manifest.json").read_text())
    u_files = [Path(path) for path in u_manifest["files"] if Path(path).name != "manifest.jsonl"]
    u_prefix = constants.STORE_PREFIX.rstrip("/") + "/" + constants.U_STORE_CELL
    u_remote = _remote_rows(
        api, [f"{u_prefix}/{p.name}" for p in u_files], revision=constants.STORE_REVISION
    )
    for path in u_files:
        records.append(
            _verify_remote(
                path, u_remote[f"{u_prefix}/{path.name}"], constants.STORE_REVISION, progress
            )
        )
    records.append(
        _stage_hf(
            api,
            {
                "path": constants.CORPUS_MANIFEST_PATH,
                "revision": constants.CORPUS_MANIFEST_REVISION,
            },
            paths["u_store"] / "manifest.jsonl",
            progress,
        )
    )
    validations.append(_validate_store(paths["u_store"], KINDS, layers))

    for prefix, dest in [
        ("issue1739_ctxmap/wildchat_rung/capture_store/wildchat", paths["wildchat"]),
        *[
            (f"issue1739_ctxmap/{rel}", dest)
            for rel, dest in zip(OOD_PREFIXES[behavior], paths["ood"], strict=True)
        ],
    ]:
        records.extend(_stage_loose(api, prefix, dest, layers, progress))
        validations.append(_validate_store(dest, KINDS, layers))

    paths.update(_stage_metadata(api, config, metadata_root, progress, records))
    manifest_path = roots["out_root"] / "staging_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "version": 1,
            "source_sha": source_sha,
            "behavior": behavior,
            "layers": layers,
            "capture_revision": CAPTURE_REVISION,
            "files": records,
            "stores": validations,
            "stager_sha256": _hashes(Path(__file__))[0],
            "checked_at": time.time(),
        },
    )
    paths["manifest"] = manifest_path
    progress.emit("staging-complete", status="complete", manifest=str(manifest_path))
    return paths


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--import-check"]:
        raise SystemExit("Use stage(config) from the registered CPU driver, or --import-check.")
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    import numpy  # noqa: F401
    from huggingface_hub import HfApi  # noqa: F401

    from explore_persona_space.experiments.issue_1739 import store_io  # noqa: F401
    from explore_persona_space.orchestrate import hub  # noqa: F401

    print("natural staging imports OK; no staging performed")
