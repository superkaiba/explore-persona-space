"""Persist an immutable trajectory-stage snapshot using the validated HF archive transport."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402
from scripts import context_risk_corrected_finish as transport  # noqa: E402
from scripts.context_risk_highrate_archive import check_original  # noqa: E402
from scripts.context_risk_trajectory_analyze import sha256  # noqa: E402
from scripts.issue2054_phase_a import _shard_large_jsonl_for_upload  # noqa: E402

PREFIX = "context_risk/issue2670_trajectory_stages"
SOURCES = [
    "scripts/context_risk_trajectory_archive.py",
    "scripts/context_risk_corrected_finish.py",
    "scripts/context_risk_highrate_archive.py",
    "scripts/context_risk_trajectory_analyze.py",
    "scripts/issue2054_phase_a.py",
    "src/explore_persona_space/orchestrate/hub.py",
    "src/explore_persona_space/orchestrate/env.py",
    "configs/eval/context_risk_trajectory_archive.yaml",
    "tests/test_context_risk_trajectory_archive.py",
]


def source_hashes() -> dict:
    root = Path(__file__).resolve().parent.parent
    expected = {name: sha256(root / name) for name in SOURCES}
    for name, module in {
        "scripts/context_risk_corrected_finish.py": transport,
        "scripts/context_risk_highrate_archive.py": sys.modules[check_original.__module__],
        "scripts/context_risk_trajectory_analyze.py": sys.modules[sha256.__module__],
        "scripts/issue2054_phase_a.py": sys.modules[_shard_large_jsonl_for_upload.__module__],
        "src/explore_persona_space/orchestrate/hub.py": hub,
        "src/explore_persona_space/orchestrate/env.py": sys.modules[load_dotenv.__module__],
    }.items():
        if sha256(Path(module.__file__)) != expected[name]:
            raise ValueError(f"Imported archive helper differs from bound source: {name}")
    return expected


def verify_staging(stage: Path, expected: dict) -> None:
    entries = [stage, *stage.rglob("*")]
    if any(p.is_symlink() for p in entries):
        raise ValueError("Archive staging acquired a symlink")
    actual = {str(p.relative_to(stage)): p for p in entries if p.is_file()}
    if set(actual) != set(expected):
        raise ValueError("Archive staging differs from manifest-derived file set")
    for name, path in actual.items():
        if (
            path.stat().st_size != expected[name]["size"]
            or sha256(path) != expected[name]["sha256"]
        ):
            raise ValueError(f"Archive staging bytes changed: {name}")


def validate_original(path: Path, expected: dict) -> dict:
    if path.suffix != ".npy":
        return check_original(path, expected)
    if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
        raise ValueError(f"Array bytes differ: {path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite archived activation: {path}")
    return {**expected, "parsed": "npy", "shape": list(array.shape), "dtype": str(array.dtype)}


def prepare_snapshot(stage: Path, manifest: dict) -> tuple[dict, dict, dict]:
    """Verify every original and reconstruct every text shard before transport."""
    entries = list(stage.rglob("*"))
    if any(p.is_symlink() for p in [stage, *entries]):
        raise ValueError("Snapshot must not contain symlinks")
    originals = manifest["files"]
    if not originals or {str(p.relative_to(stage)) for p in entries if p.is_file()} != {
        *originals,
        "snapshot_manifest.json",
    }:
        raise ValueError("Snapshot is not exactly its declared original file set")
    paths = []
    for name, expected in originals.items():
        path = stage / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(stage.resolve()):
            raise ValueError("Snapshot path escapes its root")
        validate_original(path, expected)
        if path.stat().st_size > 9_500_000 and path.suffix not in {".jsonl", ".npz", ".npy"}:
            raise ValueError(f"Oversized unshardable text artifact: {path}")
        if path.suffix == ".jsonl" and b"\r" in path.read_bytes():
            raise ValueError("Sharder must not normalize raw carriage returns")
        paths.append(path)
    retained = _shard_large_jsonl_for_upload(paths)
    for path in paths:
        if path in retained:
            continue
        record = json.loads(path.with_name(path.stem + ".manifest.json").read_text())
        import hashlib

        value = hashlib.sha256()
        for name in record["parts"]:
            part = path.parent / name
            if part.stat().st_size >= 9_000_000 or sha256(part) != record["sha256"][name]:
                raise ValueError("Invalid snapshot text shard")
            value.update(part.read_bytes())
        if value.hexdigest() != originals[str(path.relative_to(stage))]["sha256"]:
            raise ValueError("Snapshot text shard round trip differs")
    manifest_path = stage / "snapshot_manifest.json"
    transport_files = {
        str(p.relative_to(stage)): {"size": p.stat().st_size, "sha256": sha256(p)}
        for p in [*retained, manifest_path]
    }
    expected_tree = {**originals, **transport_files}
    verify_staging(stage, expected_tree)
    return originals, expected_tree, transport_files


def run(cfg: DictConfig) -> dict:
    started = time.monotonic()
    root, stage, readback = (Path(cfg[key]).resolve() for key in ("root", "stage", "readback"))
    if any(
        a.is_relative_to(b)
        for i, a in enumerate((root, stage, readback))
        for j, b in enumerate((root, stage, readback))
        if i != j
    ):
        raise ValueError("Run, snapshot and readback roots must be disjoint")
    if cfg.phase not in {"raw", "final"} or os.environ.get("EPM_HF_FILECOUNT_FALLBACK") != "0":
        raise ValueError("Unplanned phase or archive destination fallback enabled")
    if readback.exists():
        raise FileExistsError("Use a fresh readback directory")
    sources = source_hashes()
    review_path = Path(cfg.review)
    review_sha = sha256(review_path)
    review = json.loads(review_path.read_text())
    if review["verdict"] != "PASS" or review["archive_sources_sha256"] != sources:
        raise ValueError("Archive requires a source-bound independent PASS")
    manifest_path = stage / "snapshot_manifest.json"
    manifest_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["phase"] != cfg.phase:
        raise ValueError("Snapshot phase differs")
    originals, expected_tree, transport_files = prepare_snapshot(stage, manifest)
    expected_receipt = {
        f"{PREFIX}/{cfg.phase}/{name}": value for name, value in transport_files.items()
    }
    # This worker owns one archive invocation; the inherited transport uses these
    # explicitly assigned destination constants and never launches its own main().
    transport.ROOT = root
    transport.HF_PREFIX = PREFIX
    verify_staging(stage, expected_tree)
    receipt = transport.upload(stage, str(cfg.phase))
    if (
        receipt["prefix"] != f"{PREFIX}/{cfg.phase}"
        or not receipt["passed"]
        or receipt["files"] != expected_receipt
    ):
        raise ValueError("Archive transport returned a different destination")
    verify_staging(stage, expected_tree)
    landed = hub.stage_hub_prefix(
        receipt["repo_id"],
        receipt["prefix"],
        readback,
        repo_type="dataset",
        revision=receipt["revision"],
        max_workers=6,
    )
    if {str(p.relative_to(readback)) for p in landed} != set(receipt["files"]):
        raise ValueError("Remote downloaded file set differs")
    for name, expected in receipt["files"].items():
        path = readback / name
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Remote content changed: {name}")
    restored = {}
    for name, expected in originals.items():
        remote_name = f"{receipt['prefix']}/{name}"
        if name.endswith(".jsonl"):
            path = hub.stage_sharded_text(
                receipt["repo_id"],
                remote_name,
                readback / "reconstructed" / name,
                repo_type="dataset",
                revision=receipt["revision"],
                overwrite=True,
            )
        else:
            path = readback / remote_name
        restored[name] = validate_original(path, expected)
    verify_staging(stage, expected_tree)
    if (
        sources != source_hashes()
        or manifest_sha != sha256(manifest_path)
        or review_sha != sha256(review_path)
    ):
        raise ValueError("Archive source or manifest changed during verification")
    result = {
        "passed": True,
        "phase": str(cfg.phase),
        "revision": receipt["revision"],
        "repo_id": receipt["repo_id"],
        "prefix": receipt["prefix"],
        "url": receipt["url"],
        "sources_sha256": sources,
        "snapshot_sha256": manifest_sha,
        "original_files": restored,
        "downloaded_files": len(landed),
        "elapsed_seconds": time.monotonic() - started,
    }
    transport.write_json(root / f"{cfg.phase}_readback_receipt.json", result)
    return result


@hydra.main(
    version_base="1.3", config_path="../configs/eval", config_name="context_risk_trajectory_archive"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
