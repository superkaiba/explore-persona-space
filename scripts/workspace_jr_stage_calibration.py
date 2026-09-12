#!/usr/bin/env python3
"""Stage the exact complete native calibration from immutable worker uploads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import hf_hub_download  # noqa: E402

from explore_persona_space.analysis.workspace_gate import _check_upload  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    content_sha256,
    file_sha256,
    save_json,
)
from explore_persona_space.orchestrate.hub import retry_transient  # noqa: E402


def stage_file(upload, relative, destination):
    """Use pinned Hub bytes, preserve verified existing files, and hardlink atomically."""
    expected = upload["verified_sha256"][relative]
    if destination.exists():
        if destination.is_symlink() or file_sha256(destination) != expected:
            raise ValueError(f"Staged file differs from immutable upload: {destination}")
        return
    source = Path(
        retry_transient(
            lambda: hf_hub_download(
                repo_id=upload["repo"],
                repo_type="dataset",
                revision=upload["revision"],
                filename=f"{upload['prefix']}/{relative}",
            ),
            what="workspace_jr_stage_calibration",
        )
    )
    if file_sha256(source) != expected:
        raise ValueError(f"Downloaded bytes differ from verified upload: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source.resolve(), destination)


def main():
    """Require every final worker receipt; partial snapshots cannot satisfy staging."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.workers.read_text())
    role = plan["model_role"]
    if role not in {"primary", "comparison"}:
        raise ValueError("Unknown native calibration role")
    intervals = [(2, 32), (32, 61), (61, 90), (90, 119)] if role == "primary" else [(0, 119)]
    if len(plan["workers"]) != len(intervals):
        raise ValueError("Missing final calibration worker")
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "staging_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != plan:
        raise ValueError("Cannot resume staging from changed worker uploads")
    save_json(contract_path, plan)
    aggregate = {"schema": "workspace-jr-calibration-upload-aggregate-v1", "workers": []}
    completed = set()
    for rank, (worker, interval) in enumerate(zip(plan["workers"], intervals, strict=True)):
        upload = worker["upload"]
        _check_upload(upload)
        if (
            worker["rank"] != rank
            or worker["upload_content_sha256"] != content_sha256(upload)
            or upload["prefix"]
            != f"exploratory_workspace_jr/20260912/{role}_full_calibration/rank{rank}"
        ):
            raise ValueError("Worker rank, receipt digest or destination differs")
        folder = args.out / "workers" / f"rank{rank}"
        paths = {
            "upload": folder / "upload.json",
            "snapshot": folder / "snapshot.json",
            "terminal": folder / f"rank{rank}_exit.json",
        }
        save_json(paths["upload"], upload)
        stage_file(upload, "snapshot.json", paths["snapshot"])
        stage_file(upload, f"rank{rank}_exit.json", paths["terminal"])
        snapshot, terminal = (
            json.loads(paths[key].read_text()) for key in ("snapshot", "terminal")
        )
        expected = set(range(*interval)) | ({0, 1} if role == "primary" else set())
        if (
            snapshot["rank"] != rank
            or snapshot["rank_interval"] != list(interval)
            or snapshot["successful_complete"] is not True
            or snapshot["terminal_receipt_included"] is not True
            or snapshot["included_prompt_indices"] != sorted(expected)
            or snapshot["paired_prompt_files"] != len(expected)
            or terminal["rank"] != rank
            or (terminal["start"], terminal["stop"]) != interval
            or type(terminal["exit_code"]) is not int
            or terminal["exit_code"] != 0
            or type(terminal["finished_at_epoch"]) is not int
            or terminal["finished_at_epoch"] <= 1789230000
        ):
            raise ValueError("Worker snapshot lacks successful exact full-interval completion")
        for relative in ("calibration_tokens.json", "native_validation.json"):
            stage_file(upload, relative, args.out / relative)
        for index in sorted(expected):
            relative = f"lens_shards/prompt-{index:04d}.pt"
            stage_file(upload, relative, args.out / relative)
            completed.add(index)
            print(
                f"staged native calibration role={role} rank={rank} prompt={index} unique={len(completed)}/119",
                flush=True,
            )
        aggregate["workers"].append(
            {
                "rank": rank,
                "start": interval[0],
                "stop": interval[1],
                **{
                    key: {"file": str(path.relative_to(args.out)), "sha256": file_sha256(path)}
                    for key, path in paths.items()
                },
            }
        )
        save_json(args.out / "staging_partial.json", aggregate)
    if completed != set(range(119)):
        raise ValueError("Full native calibration coverage is incomplete")
    save_json(args.out / "calibration_upload.json", aggregate)
    save_json(
        args.out / "staging_complete.json",
        {
            "status": "complete",
            "model_role": role,
            "worker_plan_sha256": file_sha256(args.workers),
            "unique_prompts": len(completed),
            "calibration_upload_sha256": file_sha256(args.out / "calibration_upload.json"),
        },
    )


if __name__ == "__main__":
    main()
