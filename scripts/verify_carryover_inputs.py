#!/usr/bin/env python3
"""Verify plan-referenced LOCAL repo inputs are reachable in the git tree a
clone lane will materialize (#1469; the #734/#1434 class).

The Step 6a.5 first stanza (hub.verify_artifacts_exist) covers HF/WandB URLs
only. Every compute lane boots from a git materialization of the PUSHED
dispatch ref (GCE: `git clone --depth 1 --branch issue-<N>`; RunPod
bootstrap: init+fetch+reset; SLURM: materialize_branch_src), so a plan-cited
eval_results/ file that exists only on the VM — untracked, committed but
unpushed, or pushed to origin/main only after the branch was cut — is
guaranteed absent from the lane's tree.

NOT covered (residual risks — the gate reduces, not eliminates, the class):
config-file indirection (a path cited only inside a config the plan names),
runtime-constructed paths (the #1434 CONSUMER built its path at
issue1434_worker.py:571 — this gate catches the plan-text CITATION, which
existed in bare-filename form, not the construction), HF-staged data/ inputs
(WARN only; staging correctness is artifact-reuse check (h)(iii)'s territory),
direct dispatch_issue.py launches that bypass /issue Step 6a.5, and
extension-less citations. The check ref defaults to origin/issue-<N>; a lane
whose actual materialization ref differs (RunPod BOOTSTRAP_BRANCH defaults to
main) can be probed by threading --ref.

Exit codes: 0 = PASS (warns allowed), 1 = >=1 FAIL, 2 = usage / plan unreadable.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Extraction — TWO channels, both feeding one classifier (plan #1469 §4.1).
# ---------------------------------------------------------------------------

# Channel A — full-prefix repo-relative paths. Lookbehind excludes word chars,
# `/`, `-`, `.` so `explore-persona-space-data/issue.../f.json` (HF data-repo
# paths) and URL segments never match; `ood_eval_results` is its own
# alternative (the `eval_results` alternative cannot fire inside it — `_` is
# a word char).
_PATH_RE = re.compile(
    r"(?<![\w/\-.])"
    r"(?P<path>(?:eval_results|ood_eval_results|data)/"
    r"[A-Za-z0-9][\w.\-/*?\[\]{}<>]*)"
)
_GLOB_CHARS = set("*?[]{}<>$")
_TRAIL_PUNCT = ".,;:!?)'\"`"
# Skip reasons (never block): 'glob-or-template', 'dir' (trailing '/'),
# 'no-ext' (basename without '.'). Trailing prose punctuation rstripped first.

# Channel B — bare cited filenames (the #1434 incident's actual citation
# form): extension-bearing bare filenames (no `/` inside) cited in backticks
# or as standalone prose tokens.
_BARE_EXTS = r"(?:jsonl|json|pt|npz|csv|parquet)"  # carry-over input types in the corpus
_BARE_NAME_RE = re.compile(
    r"(?<![\w/\-.])(?P<name>[A-Za-z0-9][\w.\-]*\." + _BARE_EXTS + r")(?![\w/])"
)
_ISSUE_TOKEN_RE = re.compile(r"(?:issue[\s_-]?|#)(\d{2,4})(?!\d)", re.IGNORECASE)

_FIX_COMMITS = "f9f1002797 (main twin: e562685e40)"


def extract_candidate_paths(text: str) -> list[dict]:
    """Channel A: full-prefix candidates as {'path', 'skip_reason' (None = classify)}."""
    out: list[dict] = []
    seen: set[str] = set()
    for m in _PATH_RE.finditer(text):
        path = m.group("path").rstrip(_TRAIL_PUNCT)
        if not path or path in seen:
            continue
        seen.add(path)
        skip_reason = None
        if any(c in _GLOB_CHARS for c in path):
            skip_reason = "glob-or-template"
        elif path.endswith("/"):
            skip_reason = "dir"
        elif "." not in path.rsplit("/", 1)[-1]:
            skip_reason = "no-ext"
        out.append({"path": path, "skip_reason": skip_reason})
    return out


def extract_bare_names(text: str) -> list[str]:
    """Channel B: bare cited filenames (deduped, glob/template-char-filtered)."""
    names: list[str] = []
    seen: set[str] = set()
    for m in _BARE_NAME_RE.finditer(text):
        name = m.group("name")
        if name in seen or any(c in _GLOB_CHARS for c in name):
            continue
        seen.add(name)
        names.append(name)
    return names


def plan_issue_scope(text: str, issue: int) -> set[int]:
    """Issue-scope set = {this issue} | every issue-number token in the plan text."""
    scope = {issue}
    for m in _ISSUE_TOKEN_RE.finditer(text):
        scope.add(int(m.group(1)))
    return scope


def _worktree_hits(repo_root: Path, issue: int, pattern: str) -> list[Path]:
    return list(repo_root.glob(f".claude/worktrees/issue-{issue}*/{pattern}"))


def resolve_bare_name(name: str, *, repo_root: Path, issue: int, scope: set[int]) -> list[str]:
    """Resolve a bare cited filename to repo-relative paths under in-scope issue dirs.

    Globs eval_results/issue_<M>/** + ood_eval_results/issue_<M>/** under the
    repo root AND under this issue's worktree mirrors, for every M in the
    issue-scope set. `data/` trees are deliberately NOT globbed (huge trees,
    and the data class can only WARN anyway — a recorded residual).
    """
    resolved: list[str] = []
    for m in sorted(scope):
        for top in ("eval_results", "ood_eval_results"):
            pattern = f"{top}/issue_{m}/**/{name}"
            for hit in repo_root.glob(pattern):
                if hit.is_file():
                    resolved.append(hit.relative_to(repo_root).as_posix())
            for hit in _worktree_hits(repo_root, issue, pattern):
                if hit.is_file():
                    rel = hit.relative_to(repo_root)
                    # parts: ('.claude', 'worktrees', 'issue-<N>*', <top>, ...)
                    resolved.append(Path(*rel.parts[3:]).as_posix())
    return list(dict.fromkeys(resolved))


# ---------------------------------------------------------------------------
# Git probes (pure git — no tokens, no network beyond the bounded fetch).
# ---------------------------------------------------------------------------


def _git(repo_root: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def ref_exists(repo_root: Path, ref: str) -> bool:
    return _git(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode == 0


def path_in_ref(repo_root: Path, ref: str, path: str) -> bool:
    """True iff `path` is a git object reachable at `ref`'s tip tree."""
    return _git(repo_root, "cat-file", "-e", f"{ref}:{path}").returncode == 0


def resolve_check_ref(repo_root: Path, issue: int, *, fetch: bool = True) -> str:
    """The PUSHED dispatch ref: origin/issue-<N> if it verifies, else origin/main.

    fetch=True first runs a bounded `git fetch origin --quiet --no-tags <ref>`
    for issue-<N> and main, each fail-open (`|| true` semantics: staleness
    biases toward the committed-unpushed FAIL, whose push-and-rerun
    remediation self-heals it). Use fetch=False (--no-fetch) for tests/sweeps.
    """
    if fetch:
        for ref in (f"issue-{issue}", "main"):
            # Fail-open to possibly-stale refs (see docstring).
            with contextlib.suppress(subprocess.TimeoutExpired):
                _git(repo_root, "fetch", "origin", "--quiet", "--no-tags", ref, timeout=120)
    branch_ref = f"origin/issue-{issue}"
    if ref_exists(repo_root, branch_ref):
        return branch_ref
    return "origin/main"


def exists_locally(repo_root: Path, issue: int, path: str) -> bool:
    """True iff `path` exists at the repo root OR any issue-<N> worktree mirror."""
    if (repo_root / path).is_file():
        return True
    return any(hit.is_file() for hit in _worktree_hits(repo_root, issue, path))


# ---------------------------------------------------------------------------
# Classification — one decision ladder per candidate (plan #1469 §4.1).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Finding:
    path: str
    verdict: str  # pass | warn | fail | skip
    reason: str
    detail: str = ""
    channel: str = "A"


def classify(cand: dict, *, repo_root: Path, issue: int, check_ref: str) -> Finding:
    """Classify one concrete candidate path against the pushed check ref.

    Ladder: in-ref pass -> on-main-not-on-branch -> committed-unpushed ->
    local-only (fail for tracked-results, warn for gitignored data/) ->
    nowhere-visible (skip planned-output for own-issue Channel-A paths, else
    warn). The own-issue exemption applies ONLY at the nowhere-visible rung —
    the #1434 incident file was an own-issue INPUT that resolved locally, and
    it must FAIL.
    """
    path = cand["path"]
    channel = cand.get("channel", "A")
    cls = "data" if path.startswith("data/") else "tracked-results"
    fatal = "fail" if cls == "tracked-results" else "warn"  # data class never FAILs
    local_branch = f"issue-{issue}"

    if path_in_ref(repo_root, check_ref, path):
        return Finding(path, "pass", "in-ref", f"reachable at {check_ref}", channel)
    if check_ref != "origin/main" and path_in_ref(repo_root, "origin/main", path):
        return Finding(
            path,
            fatal,
            "on-main-not-on-branch",
            f"reachable on origin/main but not on the dispatch branch tip {check_ref} — "
            f"merge origin/main into {local_branch} (or rebase the branch) and push, "
            "then re-run (the file is already committed)",
            channel,
        )
    if ref_exists(repo_root, f"refs/heads/{local_branch}") and path_in_ref(
        repo_root, local_branch, path
    ):
        return Finding(
            path,
            fatal,
            "committed-unpushed",
            f"committed on the local {local_branch} tip but absent from {check_ref} — "
            "push the branch and re-run",
            channel,
        )
    if exists_locally(repo_root, issue, path):
        if cls == "data":
            return Finding(
                path,
                "warn",
                "data-local-only",
                "data/ is gitignored by design — the workload must self-build or HF-stage "
                "this input (artifact-reuse check (h))",
                channel,
            )
        return Finding(
            path,
            "fail",
            "untracked-local-only",
            f"exists on the VM but is not committed — git add + commit + push on "
            f"{local_branch} (the #1434 incident class, cf. {_FIX_COMMITS})",
            channel,
        )
    if channel == "B":
        return Finding(path, "skip", "bare-name-unresolved", "resolved path vanished", channel)
    if re.search(rf"issue[-_]?{issue}(?!\d)", path):
        return Finding(
            path,
            "skip",
            "planned-output",
            "own-issue path not present anywhere — treated as a planned output",
            channel,
        )
    return Finding(
        path,
        "warn",
        "unresolved-citation",
        "not in any ref and not on local disk — may resolve via HF/WandB (the first "
        "stanza) or be another issue's planned output; not provably fatal",
        channel,
    )


def run_check(plan_text: str, *, repo_root: Path, issue: int, check_ref: str) -> list[Finding]:
    """Extract (both channels), resolve, and classify every candidate."""
    findings: list[Finding] = []
    a_cands = extract_candidate_paths(plan_text)
    for cand in a_cands:
        if cand["skip_reason"]:
            findings.append(Finding(cand["path"], "skip", cand["skip_reason"], "", "A"))
        else:
            findings.append(
                classify(
                    {"path": cand["path"], "channel": "A"},
                    repo_root=repo_root,
                    issue=issue,
                    check_ref=check_ref,
                )
            )
    a_paths = {c["path"] for c in a_cands}
    a_basenames = {c["path"].rsplit("/", 1)[-1] for c in a_cands}
    scope = plan_issue_scope(plan_text, issue)
    for name in extract_bare_names(plan_text):
        if name in a_basenames:
            continue  # already covered by a Channel-A candidate
        resolved = resolve_bare_name(name, repo_root=repo_root, issue=issue, scope=scope)
        if not resolved:
            findings.append(
                Finding(
                    name,
                    "skip",
                    "bare-name-unresolved",
                    f"no file with this name under any in-scope issue dir (scope: {sorted(scope)})",
                    "B",
                )
            )
            continue
        for path in resolved:
            if path in a_paths:
                continue  # same file already classified via Channel A
            findings.append(
                classify(
                    {"path": path, "channel": "B"},
                    repo_root=repo_root,
                    issue=issue,
                    check_ref=check_ref,
                )
            )
    return findings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_repo_root() -> Path | None:
    proc = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return Path(proc.stdout.strip()).parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_carryover_inputs.py",
        description=(
            "Verify plan-cited local repo input files are reachable in the git tree "
            "the compute lane's clone will materialize (/issue Step 6a.5 second stanza)."
        ),
    )
    parser.add_argument("--plan", required=True, help="path to the approved plan markdown")
    parser.add_argument("--issue", type=int, required=True, help="task/issue number")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repo root (default: parent of `git rev-parse --git-common-dir`, worktree-safe)",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="check ref override (default: origin/issue-<N> if it exists, else origin/main)",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="skip the bounded `git fetch origin issue-<N> main` (tests / corpus sweeps)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="JSON findings")
    args = parser.parse_args(argv)

    repo_root = args.repo_root if args.repo_root is not None else _default_repo_root()
    if repo_root is None:
        print(
            "ERROR: cannot resolve repo root (not inside a git tree?) — pass --repo-root",
            file=sys.stderr,
        )
        return 2
    repo_root = Path(repo_root).resolve()

    plan_path = Path(args.plan)
    try:
        plan_text = plan_path.read_text(encoding="utf-8")
    except OSError as exc:
        # Fail loud, never exit 0 on an unreadable plan (same contract as a
        # missing plan in the first stanza).
        print(f"ERROR: cannot read plan {plan_path}: {exc}", file=sys.stderr)
        return 2

    check_ref = args.ref or resolve_check_ref(repo_root, args.issue, fetch=not args.no_fetch)
    findings = run_check(plan_text, repo_root=repo_root, issue=args.issue, check_ref=check_ref)
    n_fail = sum(f.verdict == "fail" for f in findings)
    n_warn = sum(f.verdict == "warn" for f in findings)

    if args.as_json:
        print(
            json.dumps(
                {
                    "plan": str(plan_path),
                    "issue": args.issue,
                    "check_ref": check_ref,
                    "n_fail": n_fail,
                    "n_warn": n_warn,
                    "findings": [dataclasses.asdict(f) for f in findings],
                },
                indent=2,
            )
        )
    else:
        for f in findings:
            line = f"[{f.verdict.upper():<4}] {f.path} reason={f.reason}"
            if f.detail:
                line += f" — {f.detail}"
            print(line)
        print(
            f"checked {len(findings)} citation(s) against {check_ref}: "
            f"{n_fail} fail / {n_warn} warn"
        )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
