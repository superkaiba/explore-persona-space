"""Pull eval results from WandB Artifacts to local eval_results/.

Usage:
    # List all available result artifacts
    python scripts/pull_results.py --list

    # Pull a specific result
    python scripts/pull_results.py --name results_evil_wrong_em_seed42

    # Pull all results (latest versions)
    python scripts/pull_results.py --all

    # Pull results newer than a date
    python scripts/pull_results.py --since 2026-04-14
"""

import argparse
from datetime import datetime
from pathlib import Path

import wandb

DEFAULT_PROJECT = "explore_persona_space"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "eval_results"


def list_results(project: str) -> list[dict]:
    """List all eval-results artifacts in the project."""
    api = wandb.Api()
    try:
        collections = api.artifact_type(type_name="eval-results", project=project).collections()
        results = []
        for collection in collections:
            for version in collection.versions():
                results.append(
                    {
                        "name": collection.name,
                        "version": version.version,
                        "created_at": str(version.created_at),
                        "size_mb": version.size / 1e6 if version.size else 0,
                        "metadata": version.metadata,
                    }
                )
        return results
    except Exception as e:
        print(f"Error listing artifacts: {e}")
        return []


def pull_result(project: str, name: str, output_dir: Path, version: str = "latest") -> bool:
    """Download a single result artifact."""
    api = wandb.Api()
    try:
        artifact = api.artifact(f"{project}/{name}:{version}", type="eval-results")

        # Determine local directory from artifact name
        # e.g. results_evil_wrong_em_seed42 -> evil_wrong_em_seed42
        local_name = name.removeprefix("results_")
        dest = output_dir / local_name

        artifact.download(root=str(dest))
        print(f"  ✓ {name}:{version} -> {dest}")
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Pull eval results from WandB")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="WandB project name")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available results")
    parser.add_argument("--name", type=str, help="Pull a specific artifact by name")
    parser.add_argument("--all", action="store_true", help="Pull all results")
    parser.add_argument("--since", type=str, help="Pull results newer than date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.list:
        results = list_results(args.project)
        if not results:
            print("No eval-results artifacts found.")
            return
        print(f"\n{'Name':<50} {'Version':<10} {'Size MB':<10} {'Created':<25}")
        print("-" * 95)
        for r in sorted(results, key=lambda x: x["created_at"], reverse=True):
            print(f"{r['name']:<50} {r['version']:<10} {r['size_mb']:<10.1f} {r['created_at']:<25}")
        print(f"\nTotal: {len(results)} artifacts")
        return

    if args.name:
        args.output.mkdir(parents=True, exist_ok=True)
        pull_result(args.project, args.name, args.output)
        return

    if args.all or args.since:
        results = list_results(args.project)
        if not results:
            print("No eval-results artifacts found.")
            return

        if args.since:
            cutoff = datetime.fromisoformat(args.since)
            results = [
                r
                for r in results
                if datetime.fromisoformat(r["created_at"].replace("Z", "+00:00").split("+")[0])
                >= cutoff
            ]

        # Deduplicate by name (keep latest version)
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x["created_at"], reverse=True):
            if r["name"] not in seen:
                seen.add(r["name"])
                unique.append(r)

        print(f"Pulling {len(unique)} result artifacts...")
        args.output.mkdir(parents=True, exist_ok=True)

        succeeded = 0
        for r in unique:
            if pull_result(args.project, r["name"], args.output):
                succeeded += 1

        print(f"\nDone: {succeeded}/{len(unique)} pulled to {args.output}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
