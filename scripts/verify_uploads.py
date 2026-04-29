#!/usr/bin/env python3
"""Verify that all experiment artifacts have been uploaded to permanent storage.

Called by the upload-verifier agent during status:uploading. Returns a JSON
report with PASS/FAIL per artifact category and permanent URLs for each.

Usage:
    # Check all artifacts for an issue
    uv run python scripts/verify_uploads.py --issue 42

    # Check with explicit artifact hints (from epm:results marker)
    uv run python scripts/verify_uploads.py \
        --issue 42 \
        --wandb-run "superkaiba/explore-persona-space/runs/abc123" \
        --hf-model "superkaiba1/explore-persona-space/issue-42-seed-42" \
        --pod pod3

    # Just check and print, no exit code (for interactive use)
    uv run python scripts/verify_uploads.py --issue 42 --no-fail
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Repos
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"


def check_hf_hub_path(repo_id: str, path_in_repo: str, repo_type: str = "model") -> dict:
    """Check if a path exists on HF Hub."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        prefix = path_in_repo.rstrip("/") + "/"
        matching = [f for f in files if f.startswith(prefix) or f == path_in_repo]
        if matching:
            url = f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"
            return {"status": "OK", "url": url, "file_count": len(matching)}
        return {"status": "MISSING", "url": "", "detail": f"No files under {path_in_repo}"}
    except Exception as e:
        return {"status": "ERROR", "url": "", "detail": str(e)}


def check_wandb_run(run_path: str) -> dict:
    """Check if a WandB run exists and is accessible."""
    try:
        import wandb

        api = wandb.Api()
        run = api.run(run_path)
        url = run.url
        return {"status": "OK", "url": url, "state": run.state}
    except Exception as e:
        return {"status": "MISSING", "url": "", "detail": str(e)}


def check_wandb_artifact(artifact_path: str) -> dict:
    """Check if a WandB artifact exists."""
    try:
        import wandb

        api = wandb.Api()
        artifact = api.artifact(artifact_path)
        url = f"https://wandb.ai/{artifact.entity}/{artifact.project}/artifacts/{artifact.type}/{artifact.name}"
        return {"status": "OK", "url": url, "size": artifact.size}
    except Exception as e:
        return {"status": "MISSING", "url": "", "detail": str(e)}


def check_git_figures(issue_num: int) -> dict:
    """Check if figures for this issue are committed to git."""
    repo_root = Path(__file__).resolve().parent.parent
    figure_dirs = list(repo_root.glob(f"figures/*issue*{issue_num}*")) + list(
        repo_root.glob(f"figures/*{issue_num}*")
    )

    if not figure_dirs:
        # Check for any figures committed recently that reference this issue
        result = subprocess.run(
            ["git", "log", "--oneline", "-5", "--", "figures/"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        return {
            "status": "WARN",
            "url": "",
            "detail": f"No figure directory matching issue {issue_num}. Recent figure commits: {result.stdout.strip() or 'none'}",
        }

    committed_files = []
    for d in figure_dirs:
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix in (".png", ".pdf", ".svg"):
                    # Check if committed
                    result = subprocess.run(
                        ["git", "ls-files", str(f.relative_to(repo_root))],
                        capture_output=True,
                        text=True,
                        cwd=repo_root,
                    )
                    if result.stdout.strip():
                        committed_files.append(str(f.relative_to(repo_root)))

    if committed_files:
        return {
            "status": "OK",
            "url": ", ".join(committed_files),
            "file_count": len(committed_files),
        }
    return {
        "status": "MISSING",
        "url": "",
        "detail": f"Figure dirs exist ({[str(d) for d in figure_dirs]}) but no committed .png/.pdf/.svg files",
    }


def check_pod_weights_cleaned(pod: str, output_dir: str) -> dict:
    """Check that local model weights have been cleaned from the pod."""
    if not pod:
        return {"status": "SKIP", "url": "", "detail": "No pod specified"}

    try:
        result = subprocess.run(
            [
                "ssh",
                pod,
                f"find {output_dir} -name '*.safetensors' -o -name 'model.safetensors.index.json' 2>/dev/null | head -5",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {
                "status": "WARN",
                "url": "",
                "detail": f"SSH failed: {result.stderr.strip()}",
            }
        remaining = result.stdout.strip()
        if remaining:
            return {
                "status": "FAIL",
                "url": "",
                "detail": f"Uncleaned weights found: {remaining}",
            }
        return {"status": "OK", "url": "", "detail": "No safetensors remaining"}
    except subprocess.TimeoutExpired:
        return {"status": "WARN", "url": "", "detail": "SSH timeout (pod may be stopped)"}
    except Exception as e:
        return {"status": "ERROR", "url": "", "detail": str(e)}


def check_eval_json(issue_num: int) -> dict:
    """Check that eval result JSONs exist locally."""
    repo_root = Path(__file__).resolve().parent.parent
    eval_dirs = list(repo_root.glob(f"eval_results/*issue*{issue_num}*")) + list(
        repo_root.glob(f"eval_results/*{issue_num}*")
    )

    json_files = []
    for d in eval_dirs:
        if d.is_dir():
            json_files.extend(d.glob("*.json"))
        elif d.suffix == ".json":
            json_files.append(d)

    if json_files:
        return {
            "status": "OK",
            "url": ", ".join(str(f.relative_to(repo_root)) for f in json_files[:5]),
            "file_count": len(json_files),
        }
    return {
        "status": "WARN",
        "url": "",
        "detail": f"No eval JSON files found matching issue {issue_num}",
    }


def run_verification(
    issue_num: int,
    experiment_type: str = "training",
    wandb_run: str | None = None,
    wandb_artifact: str | None = None,
    hf_model_path: str | None = None,
    hf_dataset_path: str | None = None,
    pod: str | None = None,
    output_dir: str = "/workspace/explore-persona-space/outputs",
) -> dict:
    """Run all verification checks and return structured report."""
    report = {
        "issue": issue_num,
        "experiment_type": experiment_type,
        "verdict": "PASS",
        "checks": {},
    }

    # 1. Eval JSON (always required)
    report["checks"]["eval_json"] = check_eval_json(issue_num)

    # 2. WandB run (always required for training)
    if wandb_run:
        report["checks"]["wandb_run"] = check_wandb_run(wandb_run)
    elif experiment_type == "training":
        report["checks"]["wandb_run"] = {
            "status": "MISSING",
            "url": "",
            "detail": "No WandB run path provided",
        }

    # 3. WandB artifact (eval results)
    if wandb_artifact:
        report["checks"]["wandb_artifact"] = check_wandb_artifact(wandb_artifact)

    # 4. HF model (training experiments)
    if experiment_type == "training":
        if hf_model_path:
            report["checks"]["hf_model"] = check_hf_hub_path(HF_MODEL_REPO, hf_model_path, "model")
        else:
            report["checks"]["hf_model"] = {
                "status": "MISSING",
                "url": "",
                "detail": "No HF model path provided (required for training experiments)",
            }

    # 5. HF dataset (if new data was generated)
    if hf_dataset_path:
        report["checks"]["hf_dataset"] = check_hf_hub_path(HF_DATA_REPO, hf_dataset_path, "dataset")

    # 6. Figures committed to git
    report["checks"]["figures"] = check_git_figures(issue_num)

    # 7. Pod weights cleaned (training experiments)
    if experiment_type == "training" and pod:
        report["checks"]["pod_cleanup"] = check_pod_weights_cleaned(pod, output_dir)

    # Compute overall verdict
    statuses = [c["status"] for c in report["checks"].values()]
    if (
        any(s == "FAIL" for s in statuses)
        or any(s == "MISSING" for s in statuses)
        or any(s == "ERROR" for s in statuses)
    ):
        report["verdict"] = "FAIL"
    elif any(s == "WARN" for s in statuses):
        report["verdict"] = "WARN"

    return report


def format_report(report: dict) -> str:
    """Format the verification report as markdown for a GitHub comment."""
    lines = [
        f"## Upload Verification — Issue #{report['issue']}",
        "",
        f"**Verdict: {report['verdict']}**",
        f"**Experiment type:** {report['experiment_type']}",
        "",
        "| Artifact | Status | URL / Detail |",
        "|----------|--------|-------------|",
    ]

    status_emoji = {
        "OK": "PASS",
        "MISSING": "FAIL",
        "FAIL": "FAIL",
        "WARN": "WARN",
        "ERROR": "ERROR",
        "SKIP": "SKIP",
    }

    for name, check in report["checks"].items():
        display_name = name.replace("_", " ").title()
        status = status_emoji.get(check["status"], check["status"])
        detail = check.get("url") or check.get("detail", "")
        if len(detail) > 80:
            detail = detail[:77] + "..."
        lines.append(f"| {display_name} | {status} | {detail} |")

    if report["verdict"] == "FAIL":
        lines.extend(
            [
                "",
                "**Missing artifacts must be uploaded before interpretation can begin.**",
            ]
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Verify experiment artifact uploads")
    parser.add_argument("--issue", type=int, required=True, help="GitHub issue number")
    parser.add_argument(
        "--type",
        choices=["training", "eval-only", "generation", "analysis"],
        default="training",
        help="Experiment type (determines which checks are required)",
    )
    parser.add_argument("--wandb-run", help="WandB run path (entity/project/runs/id)")
    parser.add_argument("--wandb-artifact", help="WandB artifact path")
    parser.add_argument("--hf-model", help="HF Hub model path within repo")
    parser.add_argument("--hf-dataset", help="HF Hub dataset path within repo")
    parser.add_argument("--pod", help="Pod name for cleanup verification")
    parser.add_argument("--output-dir", default="/workspace/explore-persona-space/outputs")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--no-fail", action="store_true", help="Don't exit with error on FAIL")

    args = parser.parse_args()

    report = run_verification(
        issue_num=args.issue,
        experiment_type=args.type,
        wandb_run=args.wandb_run,
        wandb_artifact=args.wandb_artifact,
        hf_model_path=args.hf_model,
        hf_dataset_path=args.hf_dataset,
        pod=args.pod,
        output_dir=args.output_dir,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))

    if report["verdict"] == "FAIL" and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
