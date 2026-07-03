#!/usr/bin/env python
"""Issue #778 v2 — bulk upload of the analysis_tensors_v2 bundle to the HF data repo.

Two phases (plan v8 §6.5 / Upload Policy):
  --phase pod : the GPU phase's complete output (extract/ rollout text + all-rollout
                acts + neutral/ text + acts) — uploaded BEFORE pod release. ONE bulk
                ``upload_folder`` commit per subtree (never a per-file loop — the
                #664 504-storm), then an EXACT-set fresh-listing verify.
  --phase vm  : the VM-produced artifacts (judge/, pairing/, rb_v2/,
                honest_nulls_maxdraws_v2/) as one bulk commit, then MANIFEST.json
                uploaded LAST — the manifest is task #816's consumption signal, so
                it must land only after every other v2 file verifies (plan §12
                assumption 18).

Fail-loud everywhere: a missing expected file on the fresh listing raises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

# hub env (HF_HUB_ENABLE_HF_TRANSFER) freezes at import time — load .env first.
load_dotenv()

import issue778_lib as lib  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate.hub import (  # noqa: E402
    DEFAULT_DATASET_REPO,
    list_repo_files_complete,
)

HF_PREFIX = "issue778_persona_vectors/analysis_tensors_v2"
POD_SUBDIRS = ("extract", "neutral")
VM_SUBDIRS = ("judge", "pairing", "rb_v2", "honest_nulls_maxdraws_v2")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _local_files(v2_root: Path, subdirs: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for sub in subdirs:
        d = v2_root / sub
        if not d.exists():
            continue
        # judge caches are per-run scratch (the raw JSONs carry the verdicts) —
        # exclude cache dirs from the durable bundle.
        out.extend(
            p
            for p in sorted(d.rglob("*"))
            if p.is_file() and "_cache" not in p.relative_to(v2_root).as_posix()
        )
    return out


def _bulk_upload_and_verify(api: HfApi, v2_root: Path, subdirs: tuple[str, ...]) -> dict:
    expected = _local_files(v2_root, subdirs)
    if not expected:
        raise RuntimeError(f"nothing to upload under {v2_root} for {subdirs}")
    for sub in subdirs:
        d = v2_root / sub
        if not d.exists():
            print(f"[upload-v2] {sub}/ absent locally — skipped", flush=True)
            continue
        api.upload_folder(
            folder_path=str(d),
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=f"{HF_PREFIX}/{sub}",
            ignore_patterns=["*_cache/*"],
            commit_message=f"issue778 v2rerun: {sub}/ bundle ({time.strftime('%Y-%m-%d')})",
        )
        print(f"[upload-v2] {sub}/ folder commit done", flush=True)
    # EXACT-set verify on a FRESH listing (never prefix-presence alone):
    # missing files AND extra stale files under OUR subdirs both fail — a
    # foreign/stale file inside the prefix corrupts the #816-consumed bundle
    # and must be resolved BEFORE the MANIFEST completion signal lands.
    listing = set(
        list_repo_files_complete(api, DEFAULT_DATASET_REPO, repo_type="dataset", revision="main")
    )
    expected_set = {f"{HF_PREFIX}/{p.relative_to(v2_root).as_posix()}" for p in expected}
    missing = sorted(expected_set - listing)
    if missing:
        raise RuntimeError(
            f"upload verify FAILED: {len(missing)} expected v2 files missing on the Hub "
            f"(first: {missing[:5]})"
        )
    hub_in_scope = {
        f for f in listing if any(f.startswith(f"{HF_PREFIX}/{sub}/") for sub in subdirs)
    }
    extra = sorted(hub_in_scope - expected_set)
    if extra:
        raise RuntimeError(
            f"upload verify FAILED: {len(extra)} EXTRA stale Hub files under "
            f"{HF_PREFIX}/{{{','.join(subdirs)}}} with no local counterpart "
            f"(first: {extra[:5]}) — delete them (or restore the local files) before "
            "the MANIFEST completion signal publishes."
        )
    return {
        "prefix": HF_PREFIX,
        "subdirs": list(subdirs),
        "n_files_verified": len(expected),
    }


def _tensor_shape(path: Path):
    import torch

    t = torch.load(path, weights_only=False, map_location="cpu")
    try:
        return list(t.shape)
    except AttributeError:
        return None


def build_manifest(v2_root: Path) -> dict:
    """MANIFEST.json — every v2 file with sha256 (+ shape for tensors) + the
    row-alignment contract. Task #816 waits on the HF presence of this file and
    consumes the bundle verbatim, so it is written/uploaded LAST."""
    files = {}
    for p in _local_files(v2_root, (*POD_SUBDIRS, *VM_SUBDIRS)):
        rel = p.relative_to(v2_root).as_posix()
        entry: dict = {"sha256": _sha256(p), "bytes": p.stat().st_size}
        if p.suffix == ".pt":
            entry["shape"] = _tensor_shape(p)
        files[rel] = entry
    return {
        "schema_version": 1,
        "label": "faithful-extraction-honest-nulls-rerun",
        "hf_prefix": HF_PREFIX,
        "row_alignment": (
            "extract/{trait}_rollouts.jsonl row i <-> extract/{trait}_acts_all.pt row i; "
            "pos rows 0..n/2-1 pair with neg rows n/2..n-1 on the same "
            "(pair_idx, question_idx, rollout_idx) key; pairing/{trait}_pairing.json "
            "carries the ONE boolean mask over the n/2 pairs (kept rows: pos=idx, "
            "neg=idx+n/2). neutral rollouts row i <-> neutral_*.pt row i."
        ),
        "paper_steering_layers_rb_index": {"evil": 19, "sycophancy": 19, "hallucination": 15},
        "files": files,
        "reproducibility": lib.repro_metadata(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #778 v2 bundle upload.")
    ap.add_argument("--out-root", default="data/issue_778")
    ap.add_argument("--phase", choices=["pod", "vm"], required=True)
    args = ap.parse_args()
    v2_root = Path(args.out_root) / "v2"
    api = HfApi()

    if args.phase == "pod":
        summary = _bulk_upload_and_verify(api, v2_root, POD_SUBDIRS)
        summary["reproducibility"] = lib.repro_metadata()
        summary["hf_data_repo"] = DEFAULT_DATASET_REPO
        print(json.dumps(summary))
        return

    # vm phase: VM-produced artifacts, then MANIFEST LAST.
    summary = _bulk_upload_and_verify(api, v2_root, VM_SUBDIRS)
    manifest = build_manifest(v2_root)
    manifest_path = v2_root / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo=f"{HF_PREFIX}/MANIFEST.json",
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        commit_message="issue778 v2rerun: MANIFEST.json (completion signal — uploaded LAST)",
    )
    listing = set(
        list_repo_files_complete(api, DEFAULT_DATASET_REPO, repo_type="dataset", revision="main")
    )
    if f"{HF_PREFIX}/MANIFEST.json" not in listing:
        raise RuntimeError("MANIFEST.json upload did not verify on the fresh listing")
    summary["manifest"] = f"{HF_PREFIX}/MANIFEST.json"
    summary["n_manifest_files"] = len(manifest["files"])
    summary["hf_data_repo"] = DEFAULT_DATASET_REPO
    summary["reproducibility"] = lib.repro_metadata()
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
