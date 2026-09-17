#!/usr/bin/env python3
"""Persist verified cross-model capture checkpoints and final analysis artifacts."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from hydra.core.config_store import ConfigStore  # noqa: E402

from explore_persona_space.backends.artifacts import write_completion_sentinel  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from scripts.story_persona_qwen38_artifacts import (  # noqa: E402
    inventory,
    push_results,
    verify_hub_entries,
)
from scripts.story_persona_qwen38_pilot import (  # noqa: E402
    digest,
    file_digest,
    read_checksums,
    read_manifest,
    validate_chunk,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = "20260917_v3"


@dataclass
class ArtifactConfig:
    """Explicit CLI configuration shared with the capture checkpoint callback."""

    phase: str = "checkpoint"
    output_dir: str = "???"
    model_key: str = "???"


ConfigStore.instance().store(name="story_persona_crossmodel_artifacts", node=ArtifactConfig)


def validate_analysis_files(out: Path, fingerprint: str, model_id: str) -> None:
    """Require all registered analysis blocks before allowing completion."""
    layers = {
        "Qwen/Qwen3.8-27B": [15, 31, 47, 63],
        "deepseek-ai/DeepSeek-V3.1-Base": [15, 30, 45, 60],
    }[model_id]
    required_blocks = {f"block_{layer}.json" for layer in layers}
    if {p.name for p in out.glob("block_*.json")} != required_blocks:
        raise ValueError("analysis does not have the exact four registered blocks")
    for name in [
        "analysis_complete.json",
        "summary.json",
        "raw_all_layers.json",
        *sorted(required_blocks),
    ]:
        if json.loads((out / name).read_text())["fingerprint"] != fingerprint:
            raise ValueError(f"analysis block fingerprint mismatch: {name}")
    done = json.loads((out / "analysis_complete.json").read_text())
    for name in ["summary.json", "raw_all_layers.json", *sorted(required_blocks)]:
        value = json.loads((out / name).read_text())
        if value["analysis_fingerprint"] != done["analysis_fingerprint"]:
            raise ValueError(f"analysis identity mismatch: {name}")
        if name in required_blocks and (
            value["status"] != "complete"
            or set(value["fits"])
            != {"full_bank", "fit_first_evaluate_last", "fit_last_evaluate_first"}
        ):
            raise ValueError(f"incomplete analysis folds: {name}")
    required_outputs = required_blocks | {
        "summary.json",
        "raw_all_layers.json",
        "centroids.npz",
        "selected_vectors.npz",
    }
    if set(done["outputs"]) != required_outputs:
        raise ValueError("analysis output inventory does not match planned outputs")
    for name, expected in done["outputs"].items():
        path = out / name
        if path.stat().st_size != expected["bytes"] or file_digest(path) != expected["sha256"]:
            raise ValueError(f"analysis output changed after completion: {name}")


def validate_capture(out: Path, *, final: bool) -> dict:
    """Verify immutable chunks and exact row identity before uploading any checkpoint."""
    manifest = read_manifest(out / "manifest.json")
    spec, fingerprint = manifest["spec"], manifest["fingerprint"]
    rows = json.loads((out / "rows.json").read_text())
    names = [p["id"] for p in spec["prompts"]]
    expected = {(p, q) for p in names for q in spec["question_ids"]}
    actual = [(r["persona"], r["question_id"]) for r in rows]
    if len(rows) != 1920 or len(expected) != 1920 or set(actual) != expected:
        raise ValueError("row identity/coverage mismatch")
    if digest(rows) != spec["inputs_sha256"]:
        raise ValueError("persisted row metadata hash mismatch")
    checksums = read_checksums(out, fingerprint)
    for k, indices in enumerate(spec["batches"]):
        path = out / "chunks" / f"batch_{k:04d}.pt"
        if path.name not in checksums:
            if path.exists() or final:
                raise ValueError(f"unverified or missing chunk: {path}")
            continue
        validate_chunk(
            path,
            fingerprint,
            indices,
            (len(indices), spec["model"]["layers"], spec["model"]["hidden_dim"]),
            expected_sha256=checksums[path.name],
        )
    if final:
        for name in ("capture_complete.json", "analysis_complete.json", "summary.json"):
            value = json.loads((out / name).read_text())
            if value["fingerprint"] != fingerprint:
                raise ValueError(f"final fingerprint mismatch: {name}")
        validate_analysis_files(out, fingerprint, spec["model"]["id"])
        if sorted(i for b in spec["batches"] for i in b) != list(range(len(rows))):
            raise ValueError("batch schedule coverage mismatch")
    return manifest


def persist(out: Path, model_key: str, *, final: bool, failed: bool) -> dict:
    """Upload a stable snapshot and verify every path, byte count and content hash."""
    if model_key not in {"qwen", "deepseek"}:
        raise ValueError("unknown model arm")
    manifest = None if failed else validate_capture(out, final=final)
    if (
        manifest is not None
        and manifest["spec"]["model"]["id"]
        != {
            "qwen": "Qwen/Qwen3.8-27B",
            "deepseek": "deepseek-ai/DeepSeek-V3.1-Base",
        }[model_key]
    ):
        raise ValueError("model arm does not match the capture")
    expected = inventory(out, include_incomplete=failed)
    suffix = f"failure_{time.time_ns()}" if failed else "analysis_tensors"
    prefix = f"issue2673_deepseek_comparison/{RUN}/{model_key}/{suffix}"
    print(f"[upload] {model_key} {len(expected)} files -> {prefix}", flush=True)
    destination = hub._upload_folder_filtered(
        out,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        prefix,
        allow_patterns=["*"],
        ignore_patterns=[
            "wandb/latest-run",
            "wandb/latest-run/**",
            "wandb/debug.log",
            "wandb/debug-internal.log",
        ],
        expected_repo_paths=[f"{prefix}/{p}" for p in expected],
        delete_after=False,
    )
    if not destination or not destination.endswith("/" + prefix):
        raise RuntimeError("upload returned no valid destination")
    repo_id = destination[: -len("/" + prefix)]
    api = HfApi()
    revision = hub._retry_upload(
        lambda: api.repo_info(repo_id, repo_type="dataset").sha,
        what="resolve cross-model checkpoint revision",
    )
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("upload lacks immutable revision")
    entries = hub._retry_upload(
        lambda: list(
            api.list_repo_tree(
                repo_id, repo_type="dataset", revision=revision, path_in_repo=prefix, recursive=True
            )
        ),
        what="verify cross-model checkpoint contents",
    )
    verify_hub_entries(entries, expected, prefix)
    if inventory(out, include_incomplete=failed) != expected:
        raise RuntimeError("artifact tree changed during upload")
    receipt = {
        "issue": 2673,
        "model_key": model_key,
        "run": RUN,
        "source_sha": os.environ["EPS_STORY_PERSONA_SOURCE_SHA"],
        "fingerprint": manifest["fingerprint"] if manifest else None,
        "verified_revision": revision,
        "hf_repo": repo_id,
        "hf_prefix": prefix,
        "repo_id": repo_id,
        "prefix": prefix,
        "files": expected,
        "file_count": len(expected),
        "total_bytes": sum(v["size"] for v in expected.values()),
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "complete": final,
        "failed_attempt": failed,
        "url": f"https://huggingface.co/datasets/{repo_id}/tree/{revision}/{prefix}",
    }
    receipt_path = out.parent / f"{out.name}_receipts" / f"receipt_{time.time_ns()}.json"
    write_json(receipt_path, receipt)
    print(f"[upload] verified {receipt['url']}", flush=True)
    return receipt


@hydra.main(version_base=None, config_path=None, config_name="story_persona_crossmodel_artifacts")
def main(cfg: ArtifactConfig) -> None:
    """Checkpoint, preserve failure evidence, or publish a fully verified model arm."""
    if cfg.phase not in {"checkpoint", "publish", "persist-failure"}:
        raise ValueError("unknown artifact phase")
    out = Path(cfg.output_dir).resolve()
    final, failed = cfg.phase == "publish", cfg.phase == "persist-failure"
    if final:
        manifest = validate_capture(out, final=True)
        if manifest["provenance"]["git_commit"] != os.environ["EPS_STORY_PERSONA_SOURCE_SHA"]:
            raise ValueError("source pin mismatch")
        master_log = Path(os.environ["EPS_STORY_MASTER_LOG"])
        shutil.copyfile(master_log, out / "workload_through_analysis.log")
    receipt = persist(out, cfg.model_key, final=final, failed=failed)
    if not final:
        return
    result_dir = ROOT / f"eval_results/issue_2673/deepseek_comparison/{cfg.model_key}"
    result_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for source in [
        out / "summary.json",
        out / "raw_all_layers.json",
        *sorted(out.glob("block_*.json")),
    ]:
        target = result_dir / source.name
        target.write_bytes(source.read_bytes())
        paths.append(target)
    write_json(result_dir / "upload_receipt.json", receipt)
    paths.append(result_dir / "upload_receipt.json")
    revision = push_results(ROOT, paths)
    completion = {**receipt, "phase": "done", "row_count": 1920, "result_git_revision": revision}
    sentinel = Path(os.environ["EPS_SENTINEL_PATH"])
    tmp = sentinel.with_suffix(".tmp")
    write_completion_sentinel(sentinel_path=tmp, issue=2673, extra=completion)
    tmp.replace(sentinel)
    write_json(
        Path("/workspace/logs") / f"issue-2673-epm_results-{time.time_ns()}.json",
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "issue": 2673,
            "task_id": 2673,
            "blocks_pipeline": False,
            "gate": "results",
            "note": completion,
        },
    )
    print(
        f"[published] {cfg.model_key} source={receipt['source_sha']} results={revision}", flush=True
    )


if __name__ == "__main__":
    main()
