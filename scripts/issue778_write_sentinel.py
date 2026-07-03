#!/usr/bin/env python
"""Issue #778 pod-side end-of-run sentinel writer.

Writes the ``poll_pipeline.py``-conforming results sentinel the orchestrator
drains (``/workspace/logs/issue-778-epm_results-<epoch>.json``), carrying the
_SENTINEL_REQUIRED_KEYS (sentinel_schema_version / kind / version) with the
marker body under ``note``. The note carries the reproducibility_card mandated
for training tasks (per-cell adapter_paths verified under hf_model_repo +
wandb_run_names/project/entity).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _wandb_entity() -> str | None:
    """Read the WandB entity off the SDK at run time (never hand-typed)."""
    try:
        import wandb

        api = wandb.Api()
        return api.default_entity
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #778 sentinel writer.")
    parser.add_argument("--issue", type=int, default=778)
    parser.add_argument("--slug", default="persona_vectors")
    parser.add_argument("--upload-summary", default="{}")
    parser.add_argument("--logs-dir", default="/workspace/logs")
    parser.add_argument("--out-root", default="data/issue_778")
    parser.add_argument("--eval-results-root", default="eval_results/issue_778")
    parser.add_argument(
        "--round",
        choices=["v1", "v2rerun"],
        default="v1",
        help="v2rerun: the faithful-extraction-honest-nulls-rerun pod phase "
        "(no training — adapter_paths/wandb are explicit N/A per plan v8 §10)",
    )
    args = parser.parse_args()

    try:
        upload_summary = json.loads(args.upload_summary)
    except json.JSONDecodeError:
        upload_summary = {"parse_error": args.upload_summary[:500]}

    exp_name = f"issue{args.issue}_{args.slug}"

    if args.round == "v2rerun":
        # The v2 rerun trains NOTHING: the 24 reused rs-LoRA adapters are
        # untouched and no WandB training run exists — stated explicitly
        # (plan v8 §10) rather than left blank.
        note = {
            "experiment": exp_name,
            "followup_label": "faithful-extraction-honest-nulls-rerun",
            "phases_completed": ["setup", "extract_v2", "neutral", "upload"],
            "note": (
                "v2 rerun pod GPU phases complete (faithful paired-mask extraction "
                "generation + ALL-rollout acts capture + neutral UltraChat capture + "
                "analysis_tensors_v2 bundle upload). The Batch-API judge phase + the "
                "paired mask + r_B v2 build + the 10-family honest null ladder run "
                "OFF-POD on the VM after pod release (plan v8 §4/§9)."
            ),
            "reproducibility_card": {
                "hf_data_repo": upload_summary.get(
                    "hf_data_repo", "superkaiba1/explore-persona-space-data"
                ),
                "analysis_tensors_v2": {
                    "prefix": upload_summary.get("prefix"),
                    "n_files_verified": upload_summary.get("n_files_verified"),
                },
                "adapter_paths": (
                    "N/A — no training this round (the 24 reused rs-LoRA adapters at "
                    "issue778_persona_vectors/adapters/ are untouched)"
                ),
                "wandb_url": (
                    "N/A — no training this round; no live training metrics exist "
                    "(generation + activation capture + closed-form statistics only)"
                ),
            },
            "reproducibility": lib.repro_metadata(),
            "eval_results_root": args.eval_results_root,
        }
        path = lib.write_results_sentinel(
            issue=args.issue,
            kind="epm:results",
            version=1,
            note=note,
            logs_dir=Path(args.logs_dir),
        )
        print(f"[sentinel] wrote {path}", flush=True)
        return
    adapters = upload_summary.get("adapters", {})
    adapter_paths = {cell: v.get("path_in_repo") for cell, v in adapters.items()}
    wandb_run_names = [f"issue778_{cell}" for cell in adapters]

    reproducibility_card = {
        "hf_model_repo": upload_summary.get("hf_model_repo"),
        "hf_data_repo": upload_summary.get("hf_data_repo"),
        "adapter_paths": adapter_paths,
        "analysis_tensors": upload_summary.get("analysis_tensors", {}),
        "wandb_project": "issue778",
        "wandb_run_names": wandb_run_names,
        "wandb_entity": _wandb_entity(),
    }

    note = {
        "experiment": exp_name,
        "phases_completed": ["setup", "extract", "monitoring", "finetune", "capture", "upload"],
        "note": (
            "Pod GPU phases (extraction + monitoring + 24 rs-LoRA finetunes + "
            "activation capture + upload) complete. The CPU null battery runs "
            "OFF-POD on the VM (plan v2 §9) against the uploaded analysis tensors."
        ),
        "reproducibility_card": reproducibility_card,
        "reproducibility": lib.repro_metadata(),
        "eval_results_root": args.eval_results_root,
    }

    path = lib.write_results_sentinel(
        issue=args.issue,
        kind="epm:results",
        version=1,
        note=note,
        logs_dir=Path(args.logs_dir),
    )
    print(f"[sentinel] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
