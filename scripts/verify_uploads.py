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

    # HEAD-verify every HF / WandB URL claimed in the epm:results marker text
    # AND the body's ## Reproducibility section (phantom-URL detection — every
    # cited URL must actually resolve at its cited revision, not just be a
    # string in a sentinel). Required for training experiments per #456.
    uv run python scripts/verify_uploads.py --issue 42 \
        --claimed-urls-file /tmp/issue-42-claimed-urls.txt

    # Just check and print, no exit code (for interactive use)
    uv run python scripts/verify_uploads.py --issue 42 --no-fail
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Make the repo's src/ importable so we can reuse the canonical HF/WandB
# HEAD-check helper (verify_artifacts_exist) instead of reimplementing it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Repos
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# Map task-workflow frontmatter ``kind`` values to the experiment type whose
# checklist rows apply when the caller omits --type. ``experiment`` stays
# "training" (the conservative default): frontmatter cannot distinguish a
# training run from an eval-only one, and silently relaxing the HF-model /
# WandB-run rows for a task that DID train would weaken the Step 8 hard
# gate. Callers that know better (the upload-verifier receives the
# experiment type as an input) pass --type explicitly (#563).
_KIND_TO_EXPERIMENT_TYPE = {
    "experiment": "training",
    "analysis": "analysis",
    "infra": "analysis",
    "batch": "analysis",
    "survey": "analysis",
}


def infer_experiment_type(issue_num: int) -> tuple[str, str]:
    """Infer the experiment type from the task's frontmatter ``kind``.

    Returns ``(experiment_type, source)``: source is ``frontmatter-kind``
    when the task's ``kind`` mapped cleanly, ``default`` when the task /
    frontmatter could not be read or the kind is unknown. Failures fall
    back to ``training`` — the STRICTEST type — so a broken inference can
    only over-demand rows, never silently relax the gate.
    """
    try:
        from explore_persona_space.task_workflow import get_task

        kind = str(get_task(issue_num)["frontmatter"].get("kind", "")).strip()
    except Exception as e:
        logger.warning(
            "could not read task %s frontmatter (%s); assuming experiment_type=training",
            issue_num,
            e,
        )
        return "training", "default"
    if kind in _KIND_TO_EXPERIMENT_TYPE:
        return _KIND_TO_EXPERIMENT_TYPE[kind], "frontmatter-kind"
    logger.warning("unknown kind %r on task %s; assuming experiment_type=training", kind, issue_num)
    return "training", "default"


def check_hf_hub_path(
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "model",
    revision: str | None = None,
) -> dict:
    """Check if a path exists on HF Hub at the given revision.

    ``revision`` defaults to ``main``. Pass a commit SHA to HEAD-verify that
    the files actually exist at the pinned revision a downstream consumer
    will dereference — this is what the phantom-URL gate needs (a string
    claiming ``/tree/<sha>/...`` is not the same as the files being there).
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)
        prefix = path_in_repo.rstrip("/") + "/"
        matching = [f for f in files if f.startswith(prefix) or f == path_in_repo]
        rev_url = revision or "main"
        if matching:
            url = f"https://huggingface.co/{repo_id}/tree/{rev_url}/{path_in_repo}"
            return {"status": "OK", "url": url, "file_count": len(matching)}
        return {
            "status": "MISSING",
            "url": "",
            "detail": f"No files under {path_in_repo} at revision {rev_url}",
        }
    except Exception as e:
        return {"status": "ERROR", "url": "", "detail": str(e)}


# Claimed-text blobs are frequently JSON (epm:results sentinels are JSON), so
# every URL is immediately followed by '",' (or '\",' when the JSON is nested).
# hub.py's _HF_URL_RE revision/path character classes exclude only '/',
# whitespace, ')' and ']', so that trailing punctuation rides into the probed
# revision/path and every HEAD check misses — a false claimed_urls FAIL
# (incident #541, 2026-06-10). Extract URL candidates permissively, strip
# trailing punctuation, and hand verify_artifacts_exist a sanitized
# one-URL-per-line view it parses cleanly.
_CLAIMED_URL_RE = re.compile(r"(?:https?|hf)://\S+")
# NOTE: '.' is deliberately NOT stripped — artifact paths legitimately end in
# '.json' / '.safetensors'; a sentence-final period stays a (pre-existing,
# rare) false MISS rather than risking real-suffix truncation.
_TRAILING_PUNCT = "\\'\",;)]}>`"


def _strip_trailing_punct(url: str) -> str:
    """Strip trailing JSON/markdown punctuation from a URL candidate.

    A trailing ``.`` is removed ONLY when the character beneath it is itself
    in the punctuation set (the markdown sentence-end case, e.g. ``` `url`. ```)
    — a period directly after a path character is kept so real suffixes like
    ``.json`` / ``.safetensors`` never truncate.
    """
    while url and (
        url[-1] in _TRAILING_PUNCT
        or (url[-1] == "." and len(url) >= 2 and url[-2] in _TRAILING_PUNCT)
    ):
        url = url[:-1].rstrip(".")
    return url


def extract_claimed_urls(text: str) -> list[str]:
    """Extract HF/WandB/hf:// URL candidates from a claimed-text blob.

    Strips trailing JSON/markdown punctuation (quotes, commas, semicolons,
    closing brackets/braces/parens, backticks, backslashes) from each match
    and de-duplicates preserving first-seen order. Returns the cleaned URLs.
    """
    return list(dict.fromkeys(_strip_trailing_punct(u) for u in _CLAIMED_URL_RE.findall(text)))


def check_claimed_urls_resolve(claimed_text_path: str | Path) -> dict:
    """HEAD-verify every HF/WandB URL claimed in a text blob actually resolves.

    The blob is typically the concatenation of the ``epm:results`` marker
    text + the body's ``## Reproducibility`` section. URLs are first
    extracted and stripped of trailing JSON/markdown punctuation (see
    ``extract_claimed_urls``), then existence-checked via
    ``explore_persona_space.orchestrate.hub.verify_artifacts_exist`` (the
    same helper /issue Step 6a.5 uses pre-launch to block on phantom
    carry-over artifacts) so behavior stays consistent at both gates.

    A claimed-but-absent URL is a hard ``FAIL`` — that is exactly the
    phantom-checkpoint condition that lets a write-up cite a file nothing
    ever uploaded. Use this BEFORE PASSing upload-verification.

    Args:
        claimed_text_path: Path to a UTF-8 text file containing the
            epm:results marker body + the Reproducibility section (and
            anything else cited). The helper scans for HF / WandB URLs;
            non-URL text is ignored.

    Returns:
        A status dict shaped like other ``check_*`` helpers.
        ``status == "OK"`` means every URL scanned resolved; ``"FAIL"``
        means one or more URLs were strings without a real artifact;
        ``"SKIP"`` means no URLs were scanned (e.g. caller did not pass
        a file); ``"ERROR"`` means a transport / auth issue propagated.
    """
    if not claimed_text_path:
        return {
            "status": "SKIP",
            "url": "",
            "detail": "No --claimed-urls-file provided",
        }
    claimed_text_path = Path(claimed_text_path)
    if not claimed_text_path.exists() or not claimed_text_path.is_file():
        return {
            "status": "ERROR",
            "url": "",
            "detail": f"claimed-urls file missing or not a file: {claimed_text_path}",
        }
    try:
        from explore_persona_space.orchestrate.hub import verify_artifacts_exist

        urls = extract_claimed_urls(claimed_text_path.read_text(encoding="utf-8"))
        # Write the sanitized one-URL-per-line view to a temp file:
        # verify_artifacts_exist takes a path and runs its own URL regexes,
        # which terminate cleanly at end-of-line once trailing punctuation
        # has been stripped here.
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".claimed-urls.txt", delete=False
        ) as tf:
            tf.write("\n".join(urls) + ("\n" if urls else ""))
            sanitized_path = Path(tf.name)
        try:
            ok, missing = verify_artifacts_exist(sanitized_path)
        finally:
            sanitized_path.unlink(missing_ok=True)
        if ok:
            return {
                "status": "OK",
                "url": str(claimed_text_path),
                "detail": "every claimed HF/WandB URL resolves at its cited revision",
            }
        return {
            "status": "FAIL",
            "url": "",
            "detail": "claimed-but-absent URLs (phantom): " + "; ".join(missing),
        }
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


def _issue_branch_ref(issue_num: int) -> str | None:
    """Return the first existing git ref for the issue branch, or None.

    Prefers the local worktree branch (``issue-<N>``) over the pushed
    remote-tracking ref (``origin/issue-<N>``). No fetch is performed —
    only refs already known to the repo are considered.
    """
    repo_root = Path(__file__).resolve().parent.parent
    for ref in (f"issue-{issue_num}", f"origin/issue-{issue_num}"):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        if result.returncode == 0:
            return ref
    return None


def issue_token_match(name: str, issue_num: int) -> bool:
    """True when ``name`` contains ``issue_num`` as a digit-bounded token.

    Substring matching is a false-PASS vector for low-numbered issues —
    issue 56 must NOT claim ``issue_563`` (or ``issue_456``) artifacts as
    its own. The number matches only when not flanked by another digit on
    either side (``issue_56`` / ``56_panel.json`` match; ``issue_563`` /
    ``2056`` do not).
    """
    return re.search(rf"(?<!\d){issue_num}(?!\d)", name) is not None


def filter_issue_paths(paths: list[str], issue_num: int) -> list[str]:
    """Keep paths whose top-level entry under the prefix names the issue.

    Mirrors the working-tree scan (``_working_tree_issue_entries``): a
    path matches when the path component directly under the prefix
    directory contains the issue number as a digit-bounded token (never
    as a substring of a longer number — see ``issue_token_match``).
    """
    return [
        p for p in paths if len(p.split("/")) >= 2 and issue_token_match(p.split("/")[1], issue_num)
    ]


def _working_tree_issue_entries(repo_root: Path, prefix: str, issue_num: int) -> list[Path]:
    """Glob working-tree entries under ``prefix`` that name the issue.

    The raw ``*<N>*`` globs substring-match (``*56*`` also hits
    ``issue_563``), so every candidate is re-checked with
    ``issue_token_match`` on its entry name before it can count as this
    issue's artifact.
    """
    candidates = list(repo_root.glob(f"{prefix}/*issue*{issue_num}*")) + list(
        repo_root.glob(f"{prefix}/*{issue_num}*")
    )
    # dict.fromkeys dedups the two-glob union (a dir matching both patterns
    # would otherwise double-count its files in the reported file_count).
    return list(dict.fromkeys(d for d in candidates if issue_token_match(d.name, issue_num)))


def _branch_files(issue_num: int, prefix: str) -> tuple[str | None, list[str]]:
    """List issue-matching files under ``prefix`` on the issue branch.

    Eval JSONs + figures are committed on the ``issue-<N>`` worktree branch
    and only reach the main working tree at the Step 9b auto-merge, so a
    working-tree-only scan false-misses mid-pipeline (#563). Returns
    ``(ref, matching_paths)``; ``(None, [])`` when no issue branch exists.
    """
    ref = _issue_branch_ref(issue_num)
    if ref is None:
        return None, []
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", prefix],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if result.returncode != 0:
        return ref, []
    return ref, filter_issue_paths(result.stdout.splitlines(), issue_num)


def check_git_figures(issue_num: int) -> dict:
    """Check if figures for this issue are committed to git.

    Scans the working tree first, then falls back to the ``issue-<N>``
    branch refs (artifacts land there before the Step 9b auto-merge).
    """
    repo_root = Path(__file__).resolve().parent.parent
    figure_dirs = _working_tree_issue_entries(repo_root, "figures", issue_num)

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

    # Not in the main working tree — scan the issue branch before
    # reporting a miss (#563: figures committed on issue-<N> pre-merge).
    ref, branch_paths = _branch_files(issue_num, "figures/")
    branch_figs = [p for p in branch_paths if p.endswith((".png", ".pdf", ".svg"))]
    if branch_figs:
        return {
            "status": "OK",
            "url": ", ".join(branch_figs[:5]),
            "file_count": len(branch_figs),
            "detail": f"committed on branch {ref}",
        }

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
            "detail": (
                f"No figure directory matching issue {issue_num} in the working tree "
                f"or on an issue-{issue_num} branch. Recent figure commits: "
                f"{result.stdout.strip() or 'none'}"
            ),
        }
    return {
        "status": "MISSING",
        "url": "",
        "detail": (
            f"Figure dirs exist ({[str(d) for d in figure_dirs]}) but no committed "
            f".png/.pdf/.svg files (working tree or issue-{issue_num} branch)"
        ),
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
    """Check that eval result JSONs exist locally or on the issue branch.

    Scans the working tree first, then falls back to the ``issue-<N>``
    branch refs (artifacts land there before the Step 9b auto-merge).
    """
    repo_root = Path(__file__).resolve().parent.parent
    eval_dirs = _working_tree_issue_entries(repo_root, "eval_results", issue_num)

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

    # Not in the main working tree — scan the issue branch before
    # reporting a miss (#563: eval JSONs committed on issue-<N> pre-merge).
    ref, branch_paths = _branch_files(issue_num, "eval_results/")
    branch_json = [p for p in branch_paths if p.endswith(".json")]
    if branch_json:
        return {
            "status": "OK",
            "url": ", ".join(branch_json[:5]),
            "file_count": len(branch_json),
            "detail": f"committed on branch {ref}",
        }
    return {
        "status": "WARN",
        "url": "",
        "detail": (
            f"No eval JSON files found matching issue {issue_num} in the working tree "
            f"or on an issue-{issue_num} branch"
        ),
    }


def run_verification(
    issue_num: int,
    experiment_type: str | None = None,
    wandb_run: str | None = None,
    wandb_artifact: str | None = None,
    hf_model_path: str | None = None,
    hf_dataset_path: str | None = None,
    pod: str | None = None,
    output_dir: str = "/workspace/explore-persona-space/outputs",
    claimed_urls_file: str | None = None,
) -> dict:
    """Run all verification checks and return structured report.

    ``experiment_type=None`` infers the type from the task's frontmatter
    ``kind`` (see ``infer_experiment_type``); an explicit value wins.
    """
    experiment_type_source = "cli"
    if experiment_type is None:
        experiment_type, experiment_type_source = infer_experiment_type(issue_num)
    report = {
        "issue": issue_num,
        "experiment_type": experiment_type,
        "experiment_type_source": experiment_type_source,
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

    # 8. Claimed-URL HEAD-check (phantom-checkpoint detection — #456).
    # Every HF/WandB URL named in the epm:results marker AND the body's
    # ## Reproducibility section MUST actually resolve at its cited revision
    # before this experiment can advance. A sentinel naming a URL string is
    # NOT evidence the underlying files exist; trusting the string is the
    # exact gap that let #456 reach awaiting_promotion with no real
    # checkpoint on HF Hub.
    report["checks"]["claimed_urls"] = check_claimed_urls_resolve(claimed_urls_file)

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
        f"**Experiment type:** {report['experiment_type']}"
        f" (source: {report.get('experiment_type_source', 'cli')})",
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
        default=None,
        help=(
            "Experiment type (determines which checks are required). Omitted: "
            "inferred from the task's frontmatter `kind` (analysis/infra/batch/"
            "survey skip the training-only rows; kind=experiment conservatively "
            "assumes training, so pass --type eval-only explicitly for eval-only "
            "experiments — #563)"
        ),
    )
    parser.add_argument("--wandb-run", help="WandB run path (entity/project/runs/id)")
    parser.add_argument("--wandb-artifact", help="WandB artifact path")
    parser.add_argument("--hf-model", help="HF Hub model path within repo")
    parser.add_argument("--hf-dataset", help="HF Hub dataset path within repo")
    parser.add_argument("--pod", help="Pod name for cleanup verification")
    parser.add_argument("--output-dir", default="/workspace/explore-persona-space/outputs")
    parser.add_argument(
        "--claimed-urls-file",
        help=(
            "Path to a text file containing the epm:results marker body + "
            "the body's ## Reproducibility section. Every HF/WandB URL in "
            "the blob is HEAD-checked against its cited revision. A "
            "claimed-but-absent URL FAILs verification (phantom-checkpoint "
            "gate — see upload-verifier.md and CLAUDE.md Gotchas #456)."
        ),
    )
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
        claimed_urls_file=args.claimed_urls_file,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))

    if report["verdict"] == "FAIL" and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
