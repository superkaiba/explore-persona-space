#!/usr/bin/env python
"""Mechanical backfill of the artifact-reuse registry from HF Hub (v2).

Enumerates the model + data repos' per-issue prefixes for a target issue set
and emits one ``artifacts/registry.jsonl`` row per (issue, repo, prefix) with
``recipe: "backfill-todo"`` — a later single Claude agent pass fills the real
recipe capsule from the task bodies. Paths only are read (never artifact
content), so this is safe against the content-hygiene rule.

Usage:
    # dry-run over an explicit issue range (default: prints rows, no write)
    uv run python scripts/backfill_artifact_registry.py --issues 800-903
    # last N issues resolved from tasks/REGISTRY.json
    uv run python scripts/backfill_artifact_registry.py --last 30
    # actually append the rows
    uv run python scripts/backfill_artifact_registry.py --last 30 --apply
    # also compute per-prefix sizes (recursive tree; slower, more HF calls)
    uv run python scripts/backfill_artifact_registry.py --last 30 --sizes

``--dry-run`` is the default; ``--apply`` appends via
``artifact_registry.append_artifact``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# scripts/ on path so `import artifact_registry` resolves in script mode and
# under test import alike.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The project dotenv wrapper MUST load before any huggingface_hub import
# (lint --check-dotenv-before-hf-import). All hf imports below are deferred
# into functions, and main() calls load_dotenv() before any of them run.
import artifact_registry

from explore_persona_space.orchestrate.env import load_dotenv

MODEL_REPO = "superkaiba1/explore-persona-space"
DATA_REPO = "superkaiba1/explore-persona-space-data"

# Shared top-level dirs that hold per-issue SUBdirs (one level down) rather
# than being per-issue themselves.
SHARED_DIRS = ("adapters", "eval_results")

# Issue-number extractors, tried in order, against a directory basename:
#   issue247 / issue-170 / issue_366 / issue458_pair_... / issue_537
#   i385_... / i432_...
_ISSUE_PATTERNS = (
    re.compile(r"^issue[_-]?(\d+)"),
    re.compile(r"^i(\d+)[_-]"),
)


def _issue_of(name: str) -> int | None:
    """Extract the issue number from a prefix/dir basename, or None."""
    base = name.rstrip("/").split("/")[-1]
    for pat in _ISSUE_PATTERNS:
        m = pat.match(base)
        if m:
            return int(m.group(1))
    return None


def _infer_type(repo_kind: str, prefix: str) -> str:
    """Best-effort artifact type from the repo + prefix name (recipe stays
    ``backfill-todo`` for the agent pass to refine). Always returns a valid
    ``artifact_registry`` type."""
    low = prefix.lower()
    if "eval_results" in low:
        return "eval-json"
    if repo_kind == "model":
        if "step_checkpoint" in low or "checkpoint" in low:
            return "checkpoint"
        # Model-repo per-issue dirs are LoRA adapters / merged dirs by project
        # default (the canonical trained artifact).
        return "adapter"
    # data repo
    if "raw_completion" in low or "completion" in low or "pool" in low:
        return "raw-completions"
    if "axis" in low or "projection" in low or "activation" in low or "store" in low:
        return "activation-store"
    if low.endswith(".jsonl"):
        return "training-mix"
    if low.endswith(".html"):
        return "dashboard"
    return "other"


def _target_issues(args: argparse.Namespace) -> set[int]:
    if args.issues:
        m = re.fullmatch(r"(\d+)-(\d+)", args.issues.strip())
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            return set(range(lo, hi + 1))
        # comma-separated fallback
        return {int(x) for x in args.issues.replace(",", " ").split()}
    # --last N from tasks/REGISTRY.json
    from explore_persona_space.task_workflow import registry_path

    reg = json.loads(Path(registry_path()).read_text(encoding="utf-8"))
    ids = sorted((int(k) for k in reg if str(k).isdigit()), reverse=True)
    return set(ids[: args.last])


def _prefix_size_bytes(api, repo_id: str, repo_type: str, prefix: str) -> int | None:
    """Sum file sizes under a prefix via a recursive tree walk, or None on any
    failure / when no sizes are reported."""
    from huggingface_hub.hf_api import RepoFile

    try:
        total = 0
        seen = False
        for e in api.list_repo_tree(
            repo_id=repo_id, repo_type=repo_type, path_in_repo=prefix, recursive=True
        ):
            if isinstance(e, RepoFile) and e.size is not None:
                total += int(e.size)
                seen = True
        return total if seen else None
    except Exception:
        return None


def _collect_prefixes(api, repo_id: str, repo_kind: str) -> list[tuple[int, str]]:
    """Return (issue, prefix) pairs for per-issue directories in a repo.

    Top-level per-issue dirs map directly; the shared ``adapters/`` and
    ``eval_results/`` dirs are descended one level to find ``issue_<N>``-style
    subdirs.
    """
    from huggingface_hub.hf_api import RepoFolder

    repo_type = "model" if repo_kind == "model" else "dataset"
    out: list[tuple[int, str]] = []
    for e in api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, recursive=False):
        if not isinstance(e, RepoFolder):
            continue
        name = e.path
        if name in SHARED_DIRS:
            for sub in api.list_repo_tree(
                repo_id=repo_id, repo_type=repo_type, path_in_repo=name, recursive=False
            ):
                if not isinstance(sub, RepoFolder):
                    continue
                issue = _issue_of(sub.path)
                if issue is not None:
                    out.append((issue, sub.path))
        else:
            issue = _issue_of(name)
            if issue is not None:
                out.append((issue, name))
    return out


def _build_rows(api, targets: set[int], compute_sizes: bool) -> list[dict]:
    rows: list[dict] = []
    for repo_id, repo_kind in ((MODEL_REPO, "model"), (DATA_REPO, "dataset")):
        repo_type = "model" if repo_kind == "model" else "dataset"
        for issue, prefix in _collect_prefixes(api, repo_id, repo_kind):
            if issue not in targets:
                continue
            size = _prefix_size_bytes(api, repo_id, repo_type, prefix) if compute_sizes else None
            rows.append(
                {
                    "id": f"{repo_kind}:{prefix}",
                    "type": _infer_type(repo_kind, prefix),
                    "path": f"{repo_id}/{prefix}",
                    "issue": issue,
                    "size_bytes": size,
                    "recipe": "backfill-todo",
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "prefix": prefix,
                    "backfilled": True,
                }
            )
    rows.sort(key=lambda r: (r["issue"], r["id"]))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill the artifact-reuse registry from HF")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--issues", help="issue range 'LO-HI' or a comma/space list")
    grp.add_argument("--last", type=int, help="the N highest issue ids from tasks/REGISTRY.json")
    parser.add_argument(
        "--sizes",
        action="store_true",
        help="compute per-prefix size_bytes via a recursive tree walk (slower)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="append the rows to the registry (default: dry-run, print only)",
    )
    parser.add_argument("--registry", default=None, help="registry path override")
    args = parser.parse_args(argv)

    load_dotenv()
    from huggingface_hub import HfApi

    api = HfApi()
    targets = _target_issues(args)
    rows = _build_rows(api, targets, compute_sizes=args.sizes)

    print(f"{len(rows)} row(s) across {len(targets)} target issue(s)", file=sys.stderr)
    if args.apply:
        for r in rows:
            artifact_registry.append_artifact(r, registry_path=args.registry)
        print(f"appended {len(rows)} row(s)", file=sys.stderr)
    else:
        for r in rows:
            print(json.dumps(r, sort_keys=True))
        print("(dry-run — pass --apply to append)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
