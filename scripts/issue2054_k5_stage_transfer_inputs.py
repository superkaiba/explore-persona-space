"""Verify and stage the existing paper banks for cross-checkpoint transfer."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explore_persona_space.orchestrate.hub import retry_transient, stage_hub_file
from scripts import issue2054_k5_loso_calibration as base
from scripts import issue2054_shared_seven as seven

SETTINGS = [
    ("Chat", "conversation_paired_stories_assistant__on_policy__chat"),
    ("Assistant-story", "conversation_paired_stories_assistant__on_policy__attrib_quoted"),
    ("HELIOS", "char_helios__on_policy__attrib_quoted"),
    ("Wren", "char_wren__on_policy__attrib_quoted"),
    ("Dana", "char_dana__on_policy__attrib_quoted"),
    ("Vex", "char_vex__on_policy__attrib_quoted"),
]
STORY_HASHES = {
    # SHA_PIN_DOMAIN: BYTES
    "qwen2.5-7b": "b872e32f8de186617fa0a1243feb36f634e3602944daf379d0a8c07017bb07bc",
    "qwen2.5-7b-instruct": seven.STORY_SHA,
}


def prepare(out, original_cache, shared_inputs):
    """Resolve pinned Hub metadata, verify local bytes and record exact own fits."""
    out.mkdir(parents=True, exist_ok=True)
    old_path = ROOT / "eval_results/issue_2054/section44_k5/k5_results.json"
    story_path = ROOT / "eval_results/issue_2054/assistant_story_k5/results.json"
    if base.sha(old_path) != seven.geometry.REFERENCE_SHA:
        raise ValueError("Original K5 result identity changed")
    # SHA_PIN_DOMAIN: BYTES
    if base.sha(story_path) != "d9f67bd2dfaf0cbd14a97af3234adb943e82bdf6b6150071af2b061817fe8070":
        raise ValueError("Assistant-story result identity changed")
    old = json.loads(old_path.read_text())
    story = json.loads(story_path.read_text())
    own = {
        r["cell"]: r["folds"]
        for r in old["results"]
        if r["k_rollouts"] == 5 and r["cohort"] == "all"
    }
    story_own = {r["model"]: r["own_folds"] for r in story["models"]}
    cached = {r["path"]: r["local"] for r in json.loads(original_cache.read_text())}
    cached.update({r["path"]: r["local"] for r in json.loads(shared_inputs.read_text())})
    settings = []
    api = HfApi()
    for label, prefix in SETTINGS:
        setting = {"label": label}
        for side, model in zip(("source", "target"), base.MODELS, strict=True):
            cell = f"{prefix}__{model}"
            if label == "Assistant-story":
                record = {
                    "path": f"issue2054_assistant_story_k5/production_v1/k5/{cell}.npz",
                    "revision": seven.STORY_REV,
                    "sha256": STORY_HASHES[model],
                }
                folds = [
                    {
                        "fold": f["fold"],
                        "n_train": f["ridge"]["n_train"],
                        "n_test": f["n_test"],
                        "ridge": f["ridge"],
                        "metrics": {"own": f["own"]},
                        "train_ids_sha256": f["train_ids_sha256"],
                        "test_ids_sha256": f["test_ids_sha256"],
                    }
                    for f in story_own[model]
                ]
            else:
                record = {
                    k: story["provenance"]["banks"][cell][k] for k in ("path", "revision", "sha256")
                }
                folds = own[cell]
            entries = retry_transient(
                lambda: api.get_paths_info(
                    base.HF_REPO, [record["path"]], repo_type="dataset", revision=record["revision"]
                ),
                what=f"verify pinned bank {cell}",
            )
            if (
                len(entries) != 1
                or entries[0].path != record["path"]
                or entries[0].lfs.sha256 != record["sha256"]
            ):
                raise ValueError(f"Pinned bank metadata mismatch: {cell}")
            local = (
                Path(cached[record["path"]])
                if record["path"] in cached
                else out / "inputs" / f"{cell}.npz"
            )
            if not local.exists():
                stage_hub_file(
                    base.HF_REPO,
                    record["path"],
                    local,
                    revision=record["revision"],
                    size_bytes=entries[0].size,
                )
            if local.stat().st_size != entries[0].size or base.sha(local) != record["sha256"]:
                raise ValueError(f"Local bank identity mismatch: {cell}")
            setting[side] = {
                **record,
                "cell": cell,
                "local": str(local),
                "own_folds": folds,
                "size": entries[0].size,
            }
            print(f"verified {label} {model} {entries[0].size} bytes", flush=True)
        settings.append(setting)
    raw_folds = subprocess.check_output(
        ["git", "show", f"{base.SOURCE_SHA}:eval_results/issue_2054/shared_fold_map.json"], cwd=ROOT
    )
    if hashlib.sha256(raw_folds).hexdigest() != seven.FOLD_SHA:
        raise ValueError("Original fold-map bytes changed")
    folds = json.loads(raw_folds)
    if folds["k"] != 5 or folds["seed"] != 137:
        raise ValueError("Unexpected original folds")
    manifest = {
        "fold_map": folds["fold_of"],
        "fold_sha256": seven.FOLD_SHA,
        "settings": settings,
        "provenance": {
            "task": 2054,
            "old_results_sha256": base.sha(old_path),
            "story_results_sha256": base.sha(story_path),
            "fold_source_sha": base.SOURCE_SHA,
            "input_metadata": "Pinned Hub LFS SHA256 and size verified against each local file",
            "cohort": "Original complete-five cohorts, caps retained; no extra quality filter",
        },
    }
    base.atomic_json(out / "manifest.json", manifest)
    return manifest


def main():
    """Build the reproducible, hash-verified input manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--original-cache", type=Path, required=True)
    parser.add_argument("--shared-inputs", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out_root, args.original_cache, args.shared_inputs)


if __name__ == "__main__":
    main()
