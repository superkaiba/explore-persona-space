#!/usr/bin/env python3
"""Issue #634: write the authoritative end-of-run epm:results sentinel.

Called by ``scripts/issue634_dispatch.sh`` AFTER Phase 1 extraction + HF upload
complete, so the sentinel carries the reproducibility card (per-cell adapter
paths N/A here — extraction-only; HF data paths + wandb_project ARE required).
Reads the extraction manifest for provenance (model, seed, layers, sampled
question indices hash, upload repo + path).

The sentinel conforms to ``poll_pipeline.py::_SENTINEL_REQUIRED_KEYS``
(``sentinel_schema_version``, ``kind``, ``version``) and the marker body goes
under ``note``. Pod-side code NEVER shells out to task.py — this writes a file
the VM orchestrator's poll loop drains.

Usage (from the dispatch script)::

    uv run python scripts/issue634_write_sentinel.py \\
        --manifest data/issue634/behavior_vectors/extraction_manifest.json \\
        --kind epm:results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

SENTINEL_SCHEMA_VERSION = 1
TASK_ID = 634
HF_PREFIX = "issue634_behavior_geometry"
WANDB_PROJECT = "issue634"


def build_card(manifest: dict) -> dict:
    """Reproducibility card for the epm:results sentinel (extraction-only)."""
    upload = manifest.get("upload", {})
    repo = upload.get("repo", "superkaiba1/explore-persona-space-data")
    path_in_repo = upload.get("path_in_repo", f"{HF_PREFIX}/analysis_tensors")
    return {
        # No LoRA / training in this analysis task — adapter_paths intentionally
        # empty (extraction-only forward passes).
        "adapter_paths": {},
        "hf_model_path": None,
        "hf_data_paths": [path_in_repo + "/"],
        "hf_data_repo": repo,
        "wandb_project": WANDB_PROJECT,
        "wandb_run_names": ["issue634-extract"],
        "model": manifest.get("model"),
        "n_roles": len(manifest.get("instance_ids", [])),
        "n_layers": manifest.get("n_layers"),
        "n_prompts": manifest.get("n_prompts"),
        "n_questions": manifest.get("n_questions"),
        "seed": manifest.get("seed"),
        "sampled_question_indices_hash": manifest.get("sampled_question_indices_hash"),
        "family_map_path": manifest.get("family_map_path"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #634: write epm:results sentinel.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kind", default="epm:results")
    parser.add_argument(
        "--eval-paths",
        nargs="*",
        default=[
            "eval_results/issue_634/joint_geometry_metrics.json",
            "eval_results/issue_634/per_layer_nn_purity.json",
            "eval_results/issue_634/coembeddability_gate.json",
            "eval_results/issue_634/cross_space_alignment.json",
            "eval_results/issue_634/panelB_nn_table.json",
        ],
        help="Phase-2 (VM-side) eval JSON paths the analyzer reads after Phase 1",
    )
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    card = build_card(manifest)

    note_obj = {
        "summary": (
            f"issue634 Phase-1 behavior-vector extraction complete: "
            f"{card['n_roles']} roles x {card['n_prompts']} prompts x "
            f"{card['n_questions']} questions, {card['n_layers']} layers, "
            f"seed={card['seed']}. Phase 2 (joint geometry) runs VM-side off the "
            f"HF-uploaded tensor + #594 context bank."
        ),
        "reproducibility_card": card,
        "eval_paths": args.eval_paths,
        "phase2_pending": True,
        "phase2_note": "VM-side: scripts/issue634_joint_geometry.py reads HF tensors",
    }

    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    kind_slug = args.kind.replace(":", "_")
    path = logs_dir / f"issue-{TASK_ID}-{kind_slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": args.kind,
        "version": 1,
        "note": json.dumps(note_obj),
        "task_id": TASK_ID,
        "by": "issue634_write_sentinel",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote sentinel {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
