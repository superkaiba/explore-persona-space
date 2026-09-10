"""Restore verified diagnostic units after GCP preemption, then continue."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k3 as k3
from scripts import issue2054_k3_diagnose as diagnostic
from explore_persona_space.orchestrate.hub import retry_transient


def main():
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--input-revision", required=True)
    parser.add_argument("--resume-revision", required=True)
    args = parser.parse_args()
    root = args.out_root.resolve()
    if root.name != "capture_diagnosis":
        raise RuntimeError("diagnostic resume requires the original diagnostic prefix")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != args.source_sha:
        raise RuntimeError("diagnostic resume driver source mismatch")
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{k3.PREFIX}/{root.name}"
    api = HfApi()
    entries = retry_transient(
        lambda: list(
            api.list_repo_tree(
                k3.HF_REPO,
                repo_type="dataset",
                revision=args.resume_revision,
                path_in_repo=prefix + "/checkpoints",
                recursive=True,
            )
        ),
        what="list interrupted diagnostic checkpoints",
    )
    remote_paths = {e.path for e in entries if hasattr(e, "size")}
    units = sorted(
        p for p in remote_paths if p.endswith(".npz") and p + ".done.json" in remote_paths
    )
    if not units:
        raise RuntimeError("no verified diagnostic units available to resume")
    files = [
        (f"{prefix}/{name}", args.input_revision)
        for name in (
            "packet.json",
            "packet.json.done.json",
            "references.npz",
            "references.npz.done.json",
        )
    ]
    files += [(p, args.resume_revision) for unit in units for p in (unit, unit + ".done.json")]

    def stage(item):
        name, revision = item
        source = k3.download(name, root, revision)
        dest = root / Path(name).relative_to(prefix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if not dest.exists() or k3.sha(dest) != k3.sha(source):
                raise RuntimeError(f"existing diagnostic staging mismatch: {dest}")
        else:
            dest.symlink_to(source.resolve())

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, job in enumerate(
            concurrent.futures.as_completed([pool.submit(stage, item) for item in files]), 1
        ):
            job.result()
            if index % 16 == 0 or index == len(files):
                k3.log(f"[phase=restore_diagnostic] staged_files={index}/{len(files)}")
    for name in ("packet.json", "references.npz"):
        if not k3.complete(root / name, "capture-diagnosis-inputs-v1"):
            raise RuntimeError("diagnostic input receipt missing")
    stores = {model: diagnostic.Checkpoints(root, model) for model in k3.MODEL_REVISIONS}
    for unit in units:
        path = root / Path(unit).relative_to(prefix)
        if not k3.complete(path, stores[path.parent.name].fingerprint):
            raise RuntimeError("diagnostic unit missing verified receipt")
    original_report_path = k3.download(
        f"{prefix}/report_qwen2.5-7b.json", root, args.resume_revision
    )
    original_report = json.loads(original_report_path.read_text())
    provenance = {
        "original_attempt": "att-20260910-214218-k3diag",
        "interruption": "GCP system preemption at2026-09-10T22:02:24Z",
        "source_revision": args.resume_revision,
        "restored_units": units,
        "resume_source_sha": actual,
        "runtime_metadata_scope": "Final report GPU/runtime/peak fields describe the resume process. Restored unit vectors and comparisons retain the original attempt provenance above; no inference recomputation is claimed for restored units.",
        "original_peak_evidence": {
            "path": f"{prefix}/report_qwen2.5-7b.json",
            "revision": args.resume_revision,
            "peak_gb": original_report["peak_gb"],
            "download_sha256": k3.sha(original_report_path),
            "qualification": "Data file uploaded before preemption; its final receipt was interrupted. Unit checkpoints are separately receipt-verified.",
        },
    }
    k3.atomic_json(root / "resume_provenance.json", provenance)
    k3.seal(root / "resume_provenance.json", root, k3.sha(__file__))
    k3.log(
        f"[phase=restore_diagnostic] verified_units={len(units)}; continuing original diagnostic"
    )
    sys.argv = [
        str(Path(diagnostic.__file__)),
        "--stage",
        "gpu",
        "--out-root",
        str(root),
        "--source-sha",
        actual,
        "--input-revision",
        args.input_revision,
    ]
    diagnostic.main()


if __name__ == "__main__":
    main()
