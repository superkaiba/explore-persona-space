#!/usr/bin/env python
"""Issue #778 corrected-monitoring-8prompt-ladder — end-of-run sentinel writer.

Writes the ``poll_pipeline.py``-conforming results sentinel the orchestrator
drains (``/workspace/logs/issue-778-epm_results-<epoch>.json``), carrying the
_SENTINEL_REQUIRED_KEYS (sentinel_schema_version / kind / version) with the marker
body under ``note``.

This amendment TRAINS NOTHING (r_B reused from #778) — so the reproducibility_card
carries NO ``adapter_paths`` and declares "no training, no wandb runs". It DOES
carry the ``eval_paths`` (the four §6.5 primary-deliverable globs) + the resolved
HF ``snapshot_download`` revision of the reused ``analysis_tensors/rb/`` +
``activations/`` fetch (consistency D2 / artifact-reuse (f)).
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
    parser = argparse.ArgumentParser(description="Issue #778 followup sentinel writer.")
    parser.add_argument("--issue", type=int, default=778)
    parser.add_argument("--slug", default="persona_vectors")
    parser.add_argument("--upload-summary", default="{}")
    parser.add_argument(
        "--reused-revision",
        default=None,
        help="resolved HF snapshot_download revision of the reused r_B/activations fetch (D2)",
    )
    parser.add_argument("--logs-dir", default="/workspace/logs")
    parser.add_argument("--eval-results-root", default="eval_results/issue_778")
    args = parser.parse_args()

    try:
        upload_summary = json.loads(args.upload_summary)
    except json.JSONDecodeError:
        upload_summary = {"parse_error": args.upload_summary[:500]}

    exp_name = f"issue{args.issue}_{args.slug}"

    # The four §6.5 primary-deliverable globs (relative to eval_results_root).
    eval_paths = [
        f"{args.eval_results_root}/{{trait}}_monitoring_corrected_nullbattery.json",
        f"{args.eval_results_root}/{{trait}}_monitoring_manyshot_nullbattery.json",
        f"{args.eval_results_root}/monitoring_corrected_{{trait}}.jsonl",
        f"{args.eval_results_root}/monitoring_manyshot_{{trait}}.jsonl",
    ]

    reproducibility_card = {
        "hf_data_repo": upload_summary.get("hf_data_repo"),
        # No training in this amendment — r_B reused from #778.
        "adapter_paths": {},
        "wandb": (
            "no training, no wandb runs (r_B reused from #778; direction-vs-null "
            "predictor re-test only)"
        ),
        "reused_artifacts": {
            "rb": f"{exp_name}/analysis_tensors/rb/{{trait}}.pt",
            "activations": f"{exp_name}/analysis_tensors/activations/{{trait}}_{{pos,neg}}.pt",
            "snapshot_revision": args.reused_revision,
        },
        "analysis_tensors_upload": upload_summary.get("analysis_tensors"),
        "exemplar_pools_upload": upload_summary.get("exemplar_pools"),
        # The primary-deliverable eval JSONLs are ALSO promoted to HF (pod cannot
        # git-commit) so the off-pod null battery can fetch them post-teardown; they
        # land in git VM-side via the /issue Step-8 eval_results/ commit.
        "eval_jsonl_upload": upload_summary.get("eval_jsonl"),
        "eval_paths": eval_paths,
    }

    note = {
        "experiment": exp_name,
        "followup_label": "corrected-monitoring-8prompt-ladder",
        "phases_completed": [
            "setup",
            "exemplar_regen",
            "monitoring_corrected",
            "monitoring_manyshot",
            "upload",
        ],
        "note": (
            "Pod GPU phases (Leg-B exemplar regen + Leg-A corrected-prompt monitoring "
            "+ Leg-B many-shot ICL monitoring + upload) complete. The CPU null battery "
            "(--input-tag monitoring_corrected / monitoring_manyshot) + null-draw upload "
            "+ figures run OFF-POD on the VM (plan v4 §9) against the uploaded tensors."
        ),
        "reproducibility_card": reproducibility_card,
        "eval_paths": eval_paths,
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
