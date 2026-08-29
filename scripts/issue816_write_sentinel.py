#!/usr/bin/env python
"""Issue #816 pod-side end-of-run sentinel writer.

Writes the ``poll_pipeline.py``-conforming results sentinel the orchestrator
drains (``/workspace/logs/issue-816-epm_results-<epoch>.json``), carrying the
_SENTINEL_REQUIRED_KEYS (sentinel_schema_version / kind / version) with the
marker body under ``note``. The note carries the reproducibility_card the upload
step already built (per-cell adapter_paths verified under hf_model_repo +
wandb_project/wandb_run_names/wandb_entity — the Step-7 training-task contract).
The Phase-B judge + Phase-C null battery run OFF-POD on the VM after this fires.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #816 sentinel writer.")
    parser.add_argument("--issue", type=int, default=816)
    parser.add_argument("--slug", default="persona_vectors")
    parser.add_argument("--upload-summary", default="{}")
    parser.add_argument("--logs-dir", default="/workspace/logs")
    parser.add_argument("--out-root", default="eval_results/issue_816/v3")
    args = parser.parse_args()

    try:
        upload_summary = json.loads(args.upload_summary)
    except json.JSONDecodeError:
        upload_summary = {"parse_error": args.upload_summary[:500]}

    exp_name = f"issue{args.issue}_{args.slug}"
    # The upload step already assembled the reproducibility_card; fall back to a
    # minimal one if the summary was unparseable (never a silent empty card).
    card = upload_summary.get("reproducibility_card") or {
        "adapter_paths": {
            cell: v.get("path_in_repo") for cell, v in upload_summary.get("adapters", {}).items()
        },
        "wandb_project": "issue816",
        "hf_model_repo": upload_summary.get("hf_model_repo"),
        "hf_data_repo": upload_summary.get("hf_data_repo"),
    }

    note = {
        "experiment": exp_name,
        "phases_completed": ["setup", "phase_a_dispatch", "upload"],
        "note": (
            "Pod Phase-A GPU work complete (Phase-0 layer probe + Exp-2 steering "
            "generation + Exp-4 preventative finetunes & post-ft eval-gen + Exp-5 "
            "activation capture + per-cell upload). The ~165k Sonnet graded-judge "
            "phase (Phase B) and the Exp-5 null-battery recompute + figures "
            "(Phase C) run OFF-POD on the VM after pod release (plan v2 §9)."
        ),
        "reproducibility_card": card,
        "reproducibility": lib.repro_metadata(),
        "eval_results_root": args.out_root,
        "analysis_tensors": upload_summary.get("analysis_tensors", {}),
        "raw_completions": upload_summary.get("raw_completions", {}),
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
