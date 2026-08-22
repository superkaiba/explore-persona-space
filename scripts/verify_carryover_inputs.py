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
extension-less citations. The check ref resolves via a precedence ladder
mirroring dispatch_issue.py's own --repo-branch resolution (#2263): an
explicit --ref verbatim > --repo-branch <b> -> origin/<b> (refusing when the
ref is absent from origin) > the invoking cwd's issue-scoped worktree branch
> the legacy origin/issue-<N>-or-origin/main default, which now REFUSES when
suffixed origin/issue-<N>-* siblings exist (the #1336 wrong-tree class). A
lane whose materialization ref is known to differ (RunPod BOOTSTRAP_BRANCH
defaults to main) is probed by threading --ref / --repo-branch accordingly.

Lane-aware (#1835): the SLURM lanes materialize via an RSYNC of
RSYNC_INCLUDE_PATHS (eval_results/ excluded), not a git clone, so
git-reachability is necessary but NOT sufficient there — under `--lane rsync`
an in-ref citation NOT covered by RSYNC_INCLUDE_PATHS + `--extra-sync-path`
downgrades to FAIL(rsync-lane-not-synced), remediated by re-dispatching with
the covering `--extra-sync-path` on BOTH this gate and `dispatch_issue.py
launch`. Include-tree membership is necessary but NOT sufficient (#1915):
the main rsync threads `--exclude <pat>` per RSYNC_EXCLUDE_PATTERNS entry
and rsync matches slash-free patterns at EVERY depth, so an in-ref citation
nested under an excluded directory name inside an include tree ALSO
downgrades — unless covered by `--extra-sync-path`, whose separate rsync
(`build_extra_rsync_command`) applies no excludes. The default
`--lane clone` is byte-identical to pre-#1835 behavior.

Plan-declared outputs (#1935): a plan's OWN structurally-declared output
files are not carry-over inputs. `extract_declared_outputs` collects fnmatch
PATTERNS (path-bearing declarations, brace-globs expanded) plus path-less
declared BASENAMES from STRUCTURED declarations only — `outputs: [...]`
bracket lists, `glob:` rows, and `- path:` rows whose nearest preceding key
line is output-semantic (`outputs:` / upload / deliverable / persist) or a
`reads:`-context row whose own list item's `produced_by:` names an INTRA-RUN
producer (`P<k>` / `(pod)` / `(vm)` / "this run"); prose mentions are never
parsed. An own-issue Channel-A candidate matching a declared pattern skips
as `planned-output-declared` (the existence-independent extension of the
nowhere-visible `planned-output` rung — post-run re-gates otherwise
false-FAIL on the plan's own outputs); an own-issue Channel-B resolution
matching a declared pattern or path-less basename skips as
`bare-name-declared-output`; a Channel-B bare-name resolution under a
FOREIGN issue dir is not a provable carry-over input and demotes to ONE
summarized `bare-name-foreign-issue` WARN per name (cite the full
repo-relative path if it IS a consumed input). Foreign Channel-A full-path
citations NEVER take the declared skip, and undeclared own-issue resolutions
keep the full ladder (the #1434 protection). Residual risk
(declared-output-and-separately-dispatched-consumer): a plan that declares X
as an output AND consumes a PRIOR run's committed copy of X (a partial/tail
dispatch of only the consuming phase) is skip-classified — the skip row
names the matched declaration for audit, and the `--extra-sync-path`
remediation remains available.

Exit codes: 0 = PASS (warns allowed), 1 = >=1 FAIL, 2 = usage / plan unreadable.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Extraction — TWO channels, both feeding one classifier (plan #1469 §4.1).
# ---------------------------------------------------------------------------

# Channel A — full-prefix repo-relative paths. Lookbehind excludes word chars,
# `/`, `-`, `.` so `explore-persona-space-data/issue.../f.json` (HF data-repo
# paths) and URL segments never match; `ood_eval_results` is its own
# alternative (the `eval_results` alternative cannot fire inside it — `_` is
# a word char).
#
# #1995 — the sibling of #1915: #1915 wired the include-tree ↔ exclude-name
# rsync-lane downgrade (`apply_rsync_lane_downgrade`) so a `tests/`/`scripts/`/
# `configs/` path nested under an `eval_results/`-style exclude classifies
# `FAIL(rsync-lane-not-synced)` — but ONLY for callers who constructed the
# `Finding` themselves. Channel A's regex only matched the three pre-existing
# prefixes, so a plan citing `tests/fixtures/eval_results/a.json` never entered
# the ladder in the first place; #1915's exclude-subtraction logic was
# reachable only via hand-built Findings. Widening the alternation to
# `(eval_results|ood_eval_results|data|tests|scripts|configs)` wires the
# extraction ↔ downgrade parity: plan text under any include-tree prefix now
# reaches `classify()` and (under `--lane rsync`) `apply_rsync_lane_downgrade`
# for the exclude subtraction. `src`, `external/open-instruct`, and `data/sft`
# are OUT of scope for this round — plans routinely cite code paths under
# them by name (not as carry-over inputs), so widening those would spike
# false-fails without corresponding coverage of a real citation class.
_PATH_RE = re.compile(
    r"(?<![\w/\-.])"
    r"(?P<path>(?:eval_results|ood_eval_results|data|tests|scripts|configs)/"
    r"[A-Za-z0-9][\w.\-/*?\[\]{}<>]*)"
)
_GLOB_CHARS = set("*?[]{}<>$")
_TRAIL_PUNCT = ".,;:!?)'\"`"
# Skip reasons (never block): 'glob-or-template', 'dir' (trailing '/'),
# 'no-ext' (basename without '.'), 'planned-output-declared' (own-issue
# Channel-A candidate matching a declared output, #1935),
# 'bare-name-declared-output' (own-issue Channel-B resolution matching a
# declared output, #1935), 'deduped-channel-a', 'bare-name-unresolved'.
# Warn reason 'bare-name-foreign-issue' (#1935) summarizes a bare name's
# foreign-issue resolutions. Trailing prose punctuation rstripped first.

# Channel B — bare cited filenames (the #1434 incident's actual citation
# form): extension-bearing bare filenames (no `/` inside) cited in backticks
# or as standalone prose tokens.
_BARE_EXTS = r"(?:jsonl|json|pt|npz|csv|parquet)"  # carry-over input types in the corpus
_BARE_NAME_RE = re.compile(
    r"(?<![\w/\-.])(?P<name>[A-Za-z0-9][\w.\-]*\." + _BARE_EXTS + r")(?![\w/])"
)
# HF-repo citations that carry a bare basename that ALSO matches _BARE_NAME_RE
# somewhere in the plan text. Two prefix families:
#   (1) short HF prefix — `issue<N>_<slug>/…/<name>.<ext>` (#1979)
#   (2) full data-repo prefix — `explore-persona-space-data/issue<N>_<slug>/…/<name>.<ext>`
# The `(?![\w/])` right guard forbids a trailing word char or `/`, mirroring
# _BARE_NAME_RE, so a longer basename (`f.jsonl.gz`) can't shadow a shorter
# hit. `_ISSUE_UNDERSCORE_GIT_PATH_GUARD` (below) protects the alt-family from
# eating repo-relative `issue_<N>/…` git paths that Channel-A handles.
_HF_CITED_RE = re.compile(
    r"(?:issue\d+_[\w\-]+|explore-persona-space-data/issue\d+_[\w\-]+)"
    # Zero-or-more `/segment` groups where each segment is a proper path
    # component (no slashes inside) — followed by a mandatory `/` right
    # before the name. This anchors the name to a real path boundary so
    # the greedy `[\w./\-]*` alternative can't backtrack into the middle
    # of a basename (#1982 review: `.../pool.jsonl` was truncating to
    # `l.jsonl` under the earlier greedy pattern).
    r"(?:/[\w.\-]+)*"
    r"/"
    r"(?P<name>[A-Za-z0-9][\w.\-]*\." + _BARE_EXTS + r")"
    r"(?![\w/])"
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


def extract_hf_cited_basenames(text: str) -> set[str]:
    """Bare basenames cited under an HF data-repo prefix (short OR full form).

    Used by run_check() to demote an `untracked-local-only` finding to a WARN
    (`hf-staged`) when the plan clearly cites the same basename via HF (#1979).
    Returns the SET of basenames only (paired-hit lookup); the caller pairs by
    basename against the Channel-B `_BARE_NAME_RE` hits.
    """
    return {m.group("name") for m in _HF_CITED_RE.finditer(text)}


def _downgrade_untracked_local_only(pending: list[Finding], name: str, hf_cited: set[str]) -> None:
    """Post-classify downgrade for Channel-B untracked-local-only fails (#1982).

    Rewrites `fail`/`untracked-local-only` findings in `pending` to WARN in two
    narrow cases, preserving the #1434 protection (unmodified when neither
    evidence branch fires):

      * `hf-staged` — the bare name is cited under an HF data-repo prefix in
        the plan (`name in hf_cited`); a coincidental VM-local mirror is not
        a repro-blocker (#1979 shape).
      * `duplicate-resolution` — a sibling resolution passed the ladder as
        `in-ref`; the plan reproduces from that committed sibling (#1739).

    Precedence: `hf-staged` wins when both would apply. Mutates `pending` in
    place; returns None.
    """
    in_ref_paths = [f.path for f in pending if f.verdict == "pass" and f.reason == "in-ref"]
    if not (name in hf_cited or in_ref_paths):
        return
    for i, f in enumerate(pending):
        if f.verdict != "fail" or f.reason != "untracked-local-only":
            continue
        if name in hf_cited:
            pending[i] = Finding(
                f.path,
                "warn",
                "hf-staged",
                f"bare name {name} is cited under an HF data-repo prefix in the "
                f"plan; local-only path {f.path} is a coincidental VM-side mirror, "
                "not a repro-blocker (#1982 / #1979)",
                "B",
            )
        elif in_ref_paths:
            head = ", ".join(in_ref_paths[:3])
            more = f" (+{len(in_ref_paths) - 3} more)" if len(in_ref_paths) > 3 else ""
            pending[i] = Finding(
                f.path,
                "warn",
                "duplicate-resolution",
                f"bare name {name} resolves to multiple paths; a sibling is "
                f"in-ref ({head}{more}) so the plan can reproduce from it — "
                f"local-only path {f.path} is not a repro-blocker (#1982 / #1739)",
                "B",
            )


def plan_issue_scope(text: str, issue: int) -> set[int]:
    """Issue-scope set = {this issue} | every issue-number token in the plan text."""
    scope = {issue}
    for m in _ISSUE_TOKEN_RE.finditer(text):
        scope.add(int(m.group(1)))
    return scope


# ---------------------------------------------------------------------------
# Plan-declared outputs (#1935) — structured declarations only, never prose.
# ---------------------------------------------------------------------------

# A bare mapping key line (no value), e.g. `outputs:` / `reads:` — the context
# anchor for `- path:` rows. An optional `- ` prefix tolerates a bare-key list
# item (`- outputs:`), though the corpus shapes are plain indented keys.
_BARE_KEY_RE = re.compile(r"^\s*(?:-\s+)?(?P<key>[A-Za-z_][\w-]*):\s*$")
_OUTPUT_KEY_RE = re.compile(r"(?i)^(outputs?|deliverables?|uploads?|persists?|artifacts?)$")
_READS_KEY_RE = re.compile(r"(?i)^(reads?|inputs?)$")
_PATH_ROW_RE = re.compile(r"^\s*-\s+path:\s*(?P<val>.+?)\s*$")
_GLOB_ROW_RE = re.compile(r"(?i)^\s*(?:-\s+)?glob:\s*(?P<val>.+?)\s*$")
_OUTPUTS_BRACKET_RE = re.compile(r"(?i)^\s*(?:-\s+)?outputs?:\s*\[(?P<items>[^\]]*)\]")
_PRODUCED_BY_RE = re.compile(r"^\s*produced_by:\s*(?P<val>.+?)\s*$")
_SUBKEY_VALUE_RE = re.compile(r"^\s*[\w-]+:\s*\S")
_LIST_ITEM_RE = re.compile(r"^\s*-\s")
# Intra-run producer (critic Must-Fix 1): a phase id (`P4`), a `(pod)`/`(vm)`
# location tag, or "this run" — an OTHER-ISSUE token anywhere in the value
# (`#1739`, `issue 1739`) vetoes, so a sibling issue's phase never collects.
_INTRA_RUN_RE = re.compile(r"(?i)(\bP\d+\b|\(pod\)|\(vm\)|\bthis run\b)")
_OTHER_ISSUE_PRODUCER_RE = re.compile(r"(?i)(#\d+|issue[\s_-]?\d+)")
# Issue dir of a RESOLVED bare-name path (resolve_bare_name only globs these
# two tops, so every resolution parses).
_ISSUE_DIR_RE = re.compile(r"^(?:ood_)?eval_results/issue_(\d+)/")


def _issue_of_path(path: str) -> int | None:
    """Issue number of a repo-relative (ood_)eval_results/issue_<M>/ path, else None."""
    m = _ISSUE_DIR_RE.match(path)
    return int(m.group(1)) if m else None


def _expand_braces(s: str) -> list[str]:
    """Expand `{a,b}` brace-globs (recursively for multiple groups)."""
    m = re.search(r"\{([^{}]*)\}", s)
    if not m:
        return [s]
    head, tail = s[: m.start()], s[m.end() :]
    out: list[str] = []
    for part in m.group(1).split(","):
        out.extend(_expand_braces(head + part.strip() + tail))
    return out


def _split_bracket_items(items: str) -> list[str]:
    """Split an inline `[a, b, ...]` list on top-level commas (brace-aware)."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in items:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [s.strip() for s in out if s.strip()]


def _collect_decl(raw: str, patterns: list[str], basenames: set[str]) -> None:
    """Route one declared value: path-bearing -> fnmatch pattern(s); path-less
    -> basename (critic Must-Fix 2 — a path-bearing declaration contributes NO
    basename). Absolute (pod-side) paths and `<template>` values are dropped.
    """
    v = raw.strip().strip("`'\"").rstrip(",")
    if not v or any(c in "<>$" for c in v):
        return
    for item in _expand_braces(v):
        item = item.strip().strip("`'\"")
        if not item or item.startswith("/"):
            continue  # absolute pod paths can never match repo-relative citations
        item = item.removeprefix("./")
        if "/" in item:
            patterns.append(item)
        else:
            basenames.add(item)


def _path_row_context(lines: list[str], i: int) -> str | None:
    """Nearest preceding bare key line's key for the `- path:` row at line i.

    Scans backward through list-item and `subkey: value` lines only; a blank
    line, fence, or prose line ends the structured block (returns None).
    """
    for j in range(i - 1, -1, -1):
        line = lines[j]
        if not line.strip():
            return None
        m = _BARE_KEY_RE.match(line)
        if m:
            return m.group("key")
        if _LIST_ITEM_RE.match(line) or _SUBKEY_VALUE_RE.match(line):
            continue
        return None
    return None


def _produced_by_within_item(lines: list[str], i: int) -> str | None:
    """`produced_by:` value within the SAME `- ` list item as line i (look-ahead
    <=3 lines; a new list item / bare key / blank line ends the item)."""
    for j in range(i + 1, min(i + 4, len(lines))):
        line = lines[j]
        if not line.strip() or _LIST_ITEM_RE.match(line) or _BARE_KEY_RE.match(line):
            return None
        m = _PRODUCED_BY_RE.match(line)
        if m:
            return m.group("val")
    return None


def extract_declared_outputs(text: str) -> tuple[list[str], set[str]]:
    """Collect the plan's STRUCTURED output declarations (#1935).

    Returns (patterns, basenames): repo-relative fnmatch PATTERNS from
    path-bearing declarations (brace-globs `{a,b}` expanded) and BASENAMES
    from path-less declared names only. Collected shapes, context-gated
    (critic Must-Fix 1 — `- path:` is a context-neutral row shape):

    - `outputs: [a, b, ...]` inline bracket lists (any context);
    - `glob: <p>` rows (the §6.5 primary_deliverable shape);
    - `- path: <p>` rows whose nearest preceding bare key line is
      output-semantic (`outputs:` / upload / deliverable / persist /
      artifact), OR is `reads:`/`inputs:` AND the item's own `produced_by:`
      (look-ahead <=3 lines within the same `- ` item) names an INTRA-RUN
      producer (`P<k>` / `(pod)` / `(vm)` / "this run", case-insensitive,
      with no other-issue token) — a file produced by this run's own phases
      is not a carry-over input. External / other-issue / absent
      `produced_by:` rows are NEVER collected.

    Prose mentions are deliberately NOT parsed (too false-positive-prone).
    `sentinel:` rows are deliberately NOT collected (recorded implementer
    decision: the corpus shapes are absolute pod paths — `/workspace/logs/…`
    — which can never match a repo-relative citation, and harvesting their
    basenames would violate the path-bearing-contributes-no-basename rule).
    """
    patterns: list[str] = []
    basenames: set[str] = set()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _OUTPUTS_BRACKET_RE.match(line)
        if m:
            for item in _split_bracket_items(m.group("items")):
                _collect_decl(item, patterns, basenames)
            continue
        m = _GLOB_ROW_RE.match(line)
        if m:
            _collect_decl(m.group("val"), patterns, basenames)
            continue
        m = _PATH_ROW_RE.match(line)
        if m:
            ctx = _path_row_context(lines, i)
            collect = False
            if ctx is not None and _OUTPUT_KEY_RE.match(ctx):
                collect = True
            elif ctx is not None and _READS_KEY_RE.match(ctx):
                pb = _produced_by_within_item(lines, i)
                if pb and _INTRA_RUN_RE.search(pb) and not _OTHER_ISSUE_PRODUCER_RE.search(pb):
                    collect = True
            if collect:
                _collect_decl(m.group("val"), patterns, basenames)
    return list(dict.fromkeys(patterns)), basenames


def _match_declared(path: str, patterns: list[str]) -> str | None:
    """First declared pattern fnmatching `path` (for auditable skip details)."""
    for pat in patterns:
        if fnmatch.fnmatchcase(path, pat):
            return pat
    return None


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

#: Repository-selection env vars that would silently repoint every `git -C`
#: probe at a DIFFERENT repository than the path-named one (#2263 MF-C).
#: Convention mirror: task_workflow.py `_GIT_ENV_POISONERS`, plus
#: GIT_COMMON_DIR (it repoints the worktree/common-dir probes specifically).
_GIT_ENV_POISONERS = (
    "GIT_DIR",
    "GIT_COMMON_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
)


def _sanitized_git_env() -> dict[str, str]:
    """os.environ minus the repository-selection GIT_* overrides (#2263 MF-C).

    A caller-inherited GIT_DIR / GIT_WORK_TREE would make every `git -C`
    probe — and the worktree/branch inference built on them — read a
    DIFFERENT repository than the path selects; stripping the poisoners makes
    path selection authoritative.
    """
    env = dict(os.environ)
    for key in _GIT_ENV_POISONERS:
        env.pop(key, None)
    return env


def _git(repo_root: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run `git -C <repo_root> <args>` under a sanitized env; never raises on rc.

    The env strips `_GIT_ENV_POISONERS` (#2263 MF-C) so the `-C` path — not a
    caller-inherited GIT_DIR/GIT_WORK_TREE — selects the repository. Returns
    the CompletedProcess (check=False); TimeoutExpired propagates.
    """
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=_sanitized_git_env(),
    )


def ref_exists(repo_root: Path, ref: str) -> bool:
    return _git(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode == 0


def path_in_ref(repo_root: Path, ref: str, path: str) -> bool:
    """True iff `path` is a git object reachable at `ref`'s tip tree."""
    return _git(repo_root, "cat-file", "-e", f"{ref}:{path}").returncode == 0


class CheckRefResolutionError(RuntimeError):
    """No unambiguous check ref could be resolved (CLI: ERROR to stderr, exit 2)."""


class RepoBranchDerivationError(RuntimeError):
    """--print-repo-branch found zero or multiple live issue-scoped worktrees (exit 2)."""


class ResolvedRef(NamedTuple):
    """A resolved check ref plus WHICH precedence-ladder rung produced it (#2263)."""

    ref: str
    #: ref-flag | repo-branch-flag | worktree-branch | bare-issue-branch-default
    #: | origin-main-default
    source: str


def _issue_scoped(branch: str, issue: int) -> bool:
    """True iff `branch` is issue-<issue> or issue-<issue>-* (digit-boundary safe)."""
    return branch == f"issue-{issue}" or branch.startswith(f"issue-{issue}-")


def _fetch_branch_refs(repo_root: Path, *branches: str) -> None:
    """Bounded per-branch `git fetch origin --quiet --no-tags <branch>` calls.

    Each is fail-open (`|| true` semantics — check=False plus the sanctioned
    TimeoutExpired suppression): staleness biases toward the
    committed-unpushed FAIL, whose push-and-rerun remediation self-heals it.
    """
    for ref in branches:
        with contextlib.suppress(subprocess.TimeoutExpired):
            _git(repo_root, "fetch", "origin", "--quiet", "--no-tags", ref, timeout=120)


def _cwd_worktree_branch(repo_root: Path, invoke_cwd: Path) -> str | None:
    """The invoking cwd's checked-out branch, iff it is a work tree OF repo_root.

    Rung-3 guards (b) + (c); any failure returns None — fail CLOSED to the
    legacy rung, never infer from an unproven cwd:

    (b) repository binding — the cwd's `--git-common-dir` equals repo_root's,
        both realpath-normalized, so a foreign or nested repository can never
        steer the check ref;
    (c) `rev-parse --is-inside-work-tree` prints `true` — a `.git`-dir cwd
        reports a branch but is not a work tree (probe P4).

    Guard (a), issue scoping of the branch name, lives in the caller.
    """
    proc = _git(invoke_cwd, "branch", "--show-current")
    branch = proc.stdout.strip()
    if proc.returncode != 0 or not branch:
        return None
    cwd_common = _git(invoke_cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")
    root_common = _git(repo_root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if cwd_common.returncode != 0 or root_common.returncode != 0:
        return None
    if os.path.realpath(cwd_common.stdout.strip()) != os.path.realpath(root_common.stdout.strip()):
        return None
    inside = _git(invoke_cwd, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return None
    return branch


def _suffixed_remote_branches(repo_root: Path, issue: int) -> list[str]:
    """Sorted `origin/issue-<issue>-*` remote-tracking refs known locally.

    Enumeration is the rung-4 ambiguity input, so a failed `for-each-ref` is
    fail-CLOSED (raises) rather than silently reading as "no candidates".
    """
    proc = _git(
        repo_root,
        "for-each-ref",
        f"refs/remotes/origin/issue-{issue}-*",
        "--format=%(refname:short)",
    )
    if proc.returncode != 0:
        raise CheckRefResolutionError(
            f"cannot enumerate origin/issue-{issue}-* candidates "
            f"(for-each-ref rc={proc.returncode}): {proc.stderr.strip()}"
        )
    return sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())


def _resolve_repo_branch_rung(repo_root: Path, repo_branch: str, *, fetch: bool) -> ResolvedRef:
    """Rung 2: an explicit --repo-branch mirrors the dispatch's own value.

    Requires origin/<repo_branch> to verify (after the bounded fetch when
    fetch=True); absent raises — the dispatch would materialize a ref that
    does not exist on origin. `main` is legal (deliberate wholly-main-resident
    confirmation).
    """
    if fetch:
        _fetch_branch_refs(repo_root, repo_branch, "main")
    branch_ref = f"origin/{repo_branch}"
    if ref_exists(repo_root, branch_ref):
        return ResolvedRef(branch_ref, "repo-branch-flag")
    hint = "after fetch" if fetch else "under --no-fetch (locally-known refs only)"
    raise CheckRefResolutionError(
        f"--repo-branch {repo_branch}: {branch_ref} does not verify {hint} — the dispatch "
        f"would materialize a ref absent from origin; push the branch "
        f"(git push origin {repo_branch}) or pass the branch the launch will actually use"
    )


def _resolve_worktree_rung(repo_root: Path, branch: str, *, fetch: bool) -> ResolvedRef:
    """Rung 3: the invoking cwd's issue-scoped worktree branch (already guarded).

    Mirrors dispatch_issue.py's current-branch defaulting. An UNPUSHED
    inferred branch raises — the lane would materialize nothing from it.
    """
    if fetch:
        _fetch_branch_refs(repo_root, branch, "main")
    branch_ref = f"origin/{branch}"
    if ref_exists(repo_root, branch_ref):
        return ResolvedRef(branch_ref, "worktree-branch")
    hint = "after fetch" if fetch else "under --no-fetch"
    raise CheckRefResolutionError(
        f"worktree branch {branch} is not on origin ({branch_ref} missing {hint}) — the lane "
        f"would materialize nothing from it; push the branch (git push origin {branch}) and "
        f"re-run, or pass --repo-branch/--ref explicitly"
    )


def _resolve_default_rung(repo_root: Path, issue: int, *, fetch: bool) -> ResolvedRef:
    """Rung 4: the legacy origin/issue-<N>-or-origin/main default + ambiguity refusal.

    Fetches issue-<N> + main as before PLUS a --prune glob fetch of
    issue-<N>-* (so a deleted suffixed sibling cannot leave a stale
    refusal-triggering tracking ref); a NON-empty suffixed candidate set
    raises listing every candidate — a bare-branch guess among suffixed
    siblings grades the WRONG tree (#1336). Under fetch=False the enumeration
    reads the locally-known remote-tracking refs (possibly stale).
    """
    if fetch:
        _fetch_branch_refs(repo_root, f"issue-{issue}", "main")
        with contextlib.suppress(subprocess.TimeoutExpired):
            _git(
                repo_root,
                "fetch",
                "--prune",
                "origin",
                "--quiet",
                "--no-tags",
                f"+refs/heads/issue-{issue}-*:refs/remotes/origin/issue-{issue}-*",
                timeout=120,
            )
    suffixed = _suffixed_remote_branches(repo_root, issue)
    bare_ref = f"origin/issue-{issue}"
    bare_exists = ref_exists(repo_root, bare_ref)
    if suffixed:
        candidates = sorted(suffixed + ([bare_ref] if bare_exists else []))
        raise CheckRefResolutionError(
            f"cannot pick a check ref for issue {issue}: suffixed origin/issue-{issue}-* "
            f"branch(es) exist — candidates: {', '.join(candidates)}. A bare-branch guess "
            f"among suffixed siblings grades the WRONG tree (#1336). Remedies, in order: "
            f"(1) pass --repo-branch <branch> mirroring the launch's --repo-branch (lead "
            f"remedy; derive it via --print-repo-branch); (2) run the gate from the issue "
            f"worktree so inference picks its branch; (3) an explicit --ref LAST — caution: "
            f"--ref bypasses the ref-existence guard and can deliberately grade a tree the "
            f"dispatch will not materialize"
        )
    if bare_exists:
        return ResolvedRef(bare_ref, "bare-issue-branch-default")
    return ResolvedRef("origin/main", "origin-main-default")


def resolve_check_ref(
    repo_root: Path,
    issue: int,
    *,
    fetch: bool = True,
    repo_branch: str | None = None,
    invoke_cwd: Path | None = None,
) -> ResolvedRef:
    """Resolve the ref the dispatch will ACTUALLY materialize + its source (#2263).

    Precedence ladder (rung 1 — an explicit --ref — is handled by the CLI and
    never calls this resolver):

    2. `repo_branch` (--repo-branch): the dispatch's own value, mirrored to
       this gate — check ref origin/<repo_branch>, REFUSED when absent from
       origin after the bounded fetch. Source `repo-branch-flag`.
    3. Worktree inference: when the INVOKING cwd (`invoke_cwd`, default
       Path.cwd()) is a live work tree OF `repo_root` checked out on an
       issue-<issue>-scoped branch, that branch mirrors dispatch_issue.py's
       current-branch defaulting; an unpushed inferred branch REFUSES.
       Source `worktree-branch`.
    4. Legacy default + ambiguity refusal: origin/issue-<N> if it exists,
       else origin/main — REFUSED when suffixed origin/issue-<N>-* siblings
       exist (candidates listed). Sources `bare-issue-branch-default` /
       `origin-main-default`.

    fetch=True runs bounded fail-open per-branch fetches (see
    `_fetch_branch_refs`); fetch=False (--no-fetch) resolves against
    possibly-STALE locally-known origin/* refs — a just-pushed sibling branch
    is invisible until fetched.

    Residual divergence window (#2263 review finding 1): the gate and the
    launch resolve INDEPENDENTLY, so a worktree added/removed — or a branch
    pushed/deleted — between the two fences changes the resolution. The
    window is real and fails CLOSED in both directions (worktree added =>
    the launch fence's re-derivation hits the multiple-match refusal;
    worktree removed => the zero-match refusal). The dispatch-side conflict
    guard does NOT bound it — ANY explicit --repo-branch value bypasses that
    guard by construction (scripts/dispatch_issue.py:1203) and the pushed-ref
    ls-remote existence check only WARNs.
    """
    if repo_branch is not None:
        return _resolve_repo_branch_rung(repo_root, repo_branch, fetch=fetch)
    cwd = invoke_cwd if invoke_cwd is not None else Path.cwd()
    branch = _cwd_worktree_branch(repo_root, cwd)
    if branch is not None and _issue_scoped(branch, issue):
        return _resolve_worktree_rung(repo_root, branch, fetch=fetch)
    return _resolve_default_rung(repo_root, issue, fetch=fetch)


def derive_local_branch(check_ref: str, ref_source: str, issue: int) -> str:
    """The LOCAL branch whose unpushed tip classify() should probe (#2263 MF-3).

    Source-aware: an explicitly-resolved branch ref (`repo-branch-flag` /
    `worktree-branch`) probes ITS bare branch name (main included); an
    explicit --ref strips `origin/` ONLY when the stripped name lies in this
    issue's own namespace (issue-<N> / issue-<N>-*, digit-boundary safe),
    else falls back to issue-<N>; both defaults keep the legacy issue-<N>.
    """
    if ref_source in ("repo-branch-flag", "worktree-branch"):
        return check_ref.removeprefix("origin/")
    if ref_source == "ref-flag":
        name = check_ref.removeprefix("origin/")
        if _issue_scoped(name, issue):
            return name
    return f"issue-{issue}"


def _parse_worktree_list(porcelain: str) -> list[tuple[Path, str | None]]:
    """Parse `git worktree list --porcelain` output into
    `(worktree path, branch ref or None)` records. Records are blank-line
    separated; each opens with `worktree <path>` and carries `branch <ref>`
    only for branch-bound checkouts (a DETACHED checkout has a bare
    `detached` line instead). Pure string parsing — no git, no filesystem.

    Convention copy of scripts/verify_task_body.py `_parse_worktree_list`
    (#2288); duplicated rather than imported to avoid coupling the two
    verifiers' release cadences (#2263 plan §10).
    """
    records: list[tuple[Path, str | None]] = []
    path: Path | None = None
    branch: str | None = None
    for line in porcelain.splitlines():
        if line.startswith("worktree "):
            if path is not None:
                records.append((path, branch))
            path = Path(line[len("worktree ") :])
            branch = None
        elif line.startswith("branch "):
            branch = line[len("branch ") :]
        elif not line.strip():
            if path is not None:
                records.append((path, branch))
            path = None
            branch = None
    if path is not None:
        records.append((path, branch))
    return records


def derive_worktree_repo_branch(repo_root: Path, issue: int) -> str:
    """The ONE live issue-<issue>-scoped worktree branch (#2263 MF-A).

    The shared --repo-branch resolver both Step 6 fences call
    (--print-repo-branch). Parses `git worktree list --porcelain` (pure local
    read — no network), keeps records whose branch is issue-scoped
    (issue-<N> / issue-<N>-*, digit-boundary safe) AND whose worktree
    directory still exists (a registered-but-DELETED worktree keeps its
    branch line plus a `prunable` marker — probe P1b), and:

    - exactly ONE survivor -> returns its bare branch name (structurally
      never `main` — out of the issue namespace by construction);
    - ZERO -> raises RepoBranchDerivationError (a wholly-main-resident
      workload passes --repo-branch main explicitly at BOTH fences);
    - MULTIPLE -> raises listing every (branch, path) candidate.

    Skipped stale registrations are named in the refusal messages AND, on
    success, in one stderr note (#2263 review finding 2): a transient
    is_dir() False — e.g. a mount hiccup — can turn a designed 2-worktree
    multiple-match refusal into a unique match, and the note makes that
    visible. Diagnostic only; stdout stays the bare branch name.
    """
    proc = _git(repo_root, "worktree", "list", "--porcelain")
    if proc.returncode != 0:
        raise RepoBranchDerivationError(
            f"git worktree list --porcelain failed (rc={proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    live: list[tuple[str, Path]] = []
    stale: list[tuple[str, Path]] = []
    for wt_path, branch_ref in _parse_worktree_list(proc.stdout):
        if branch_ref is None:
            continue  # detached worktree — carries no branch to mirror
        branch = branch_ref.removeprefix("refs/heads/")
        if not _issue_scoped(branch, issue):
            continue
        if wt_path.is_dir():
            live.append((branch, wt_path))
        else:
            stale.append((branch, wt_path))
    stale_desc = ", ".join(f"{b} @ {p}" for b, p in stale)
    if len(live) == 1:
        if stale:
            print(
                f"NOTE: skipped stale issue-{issue} worktree registration(s) "
                f"(directory missing): {stale_desc} — a transient miss here can mask a "
                f"multiple-match refusal (#2263)",
                file=sys.stderr,
            )
        return live[0][0]
    if not live:
        raise RepoBranchDerivationError(
            f"no live worktree of this repository is checked out on an issue-{issue}-scoped "
            f"branch (issue-{issue} / issue-{issue}-*)"
            + (
                f"; skipped stale registration(s) (directory missing): {stale_desc}"
                if stale
                else ""
            )
            + f" — a wholly-main-resident workload passes --repo-branch main explicitly at BOTH "
            f"the gate and the launch; otherwise create the issue-{issue} worktree "
            f"(scripts/new_worktree.sh) first"
        )
    listing = "; ".join(f"{b} @ {p}" for b, p in live)
    raise RepoBranchDerivationError(
        f"multiple live issue-{issue}-scoped worktrees: {listing}"
        + (f"; skipped stale registration(s): {stale_desc}" if stale else "")
        + " — pass --repo-branch <branch> explicitly, threaded to BOTH the gate and the launch"
    )


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


def _committed_unpushed_detail(found_branch: str, check_ref: str) -> str:
    """committed-unpushed remediation, split on found-branch vs check-ref branch (#2263).

    Three wordings: (1) found == check-ref branch — today's push-and-rerun;
    (2) found != check-ref branch, origin/-shaped check ref — merge/rebase or
    re-dispatch with --repo-branch <found>; (3) non-branch-shaped check ref
    (a raw --ref SHA / local ref) — no branch is interpolated from it.
    """
    if check_ref == f"origin/{found_branch}":
        return (
            f"committed on the local {found_branch} tip but absent from {check_ref} — "
            "push the branch and re-run"
        )
    if check_ref.startswith("origin/"):
        return (
            f"committed on the local {found_branch} tip but the dispatch materializes "
            f"{check_ref} — merge/rebase {found_branch} into "
            f"{check_ref.removeprefix('origin/')} and push, or re-dispatch with "
            f"--repo-branch {found_branch}"
        )
    return (
        f"committed on the local {found_branch} tip but absent from the checked ref "
        f"{check_ref} — push {found_branch} and re-dispatch with --repo-branch "
        f"{found_branch}, or merge {found_branch} into the branch the dispatch will "
        "materialize"
    )


def classify(
    cand: dict,
    *,
    repo_root: Path,
    issue: int,
    check_ref: str,
    local_branch: str | None = None,
) -> Finding:
    """Classify one concrete candidate path against the pushed check ref.

    Ladder: in-ref pass -> on-main-not-on-branch -> committed-unpushed ->
    local-only (fail for tracked-results, warn for gitignored data/) ->
    nowhere-visible (skip planned-output for own-issue Channel-A paths, else
    warn). The own-issue exemption applies ONLY at the nowhere-visible rung —
    the #1434 incident file was an own-issue INPUT that resolved locally, and
    it must FAIL.

    `local_branch` (#2263 MF-3) is the source-aware `derive_local_branch`
    value; None keeps the legacy issue-<N>. The committed-unpushed rung
    probes the deduped UNION [local_branch, issue-<N>] (first hit wins) so a
    suffixed check ref still catches a commit stranded on the legacy bare
    branch and vice versa; no derivation may convert that FAIL to SKIP/WARN.
    """
    path = cand["path"]
    channel = cand.get("channel", "A")
    cls = "data" if path.startswith("data/") else "tracked-results"
    fatal = "fail" if cls == "tracked-results" else "warn"  # data class never FAILs
    if local_branch is None:
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
    for probe_branch in dict.fromkeys((local_branch, f"issue-{issue}")):
        if ref_exists(repo_root, f"refs/heads/{probe_branch}") and path_in_ref(
            repo_root, probe_branch, path
        ):
            return Finding(
                path,
                fatal,
                "committed-unpushed",
                _committed_unpushed_detail(probe_branch, check_ref),
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


def run_check(
    plan_text: str,
    *,
    repo_root: Path,
    issue: int,
    check_ref: str,
    local_branch: str | None = None,
) -> list[Finding]:
    """Extract (both channels), resolve, and classify every candidate.

    #2263: `local_branch` (the source-aware `derive_local_branch` value)
    threads into every classify() call; None keeps the legacy issue-<N>
    committed-unpushed probe (the #1995 sweep caller's contract).

    #1935 restructure: plan-declared outputs are computed once from the plan's
    STRUCTURED declarations (`extract_declared_outputs`). Channel A: an
    own-issue candidate fnmatching a declared pattern skips BEFORE classify
    (`planned-output-declared`); foreign candidates never take the skip.
    Channel B, per bare name: dedup-by-resolved-path first (unchanged), then a
    PER-PATH own/foreign partition — an own-issue resolution matching a
    declared pattern / path-less declared basename skips
    (`bare-name-declared-output`), an undeclared own-issue resolution keeps
    the FULL ladder (the #1434 protection), and foreign-issue resolutions
    collapse into ONE summarized `bare-name-foreign-issue` WARN per name.
    """
    findings: list[Finding] = []
    declared_patterns, declared_basenames = extract_declared_outputs(plan_text)
    own_issue_re = re.compile(rf"issue[-_]?{issue}(?!\d)")
    a_cands = extract_candidate_paths(plan_text)
    for cand in a_cands:
        if cand["skip_reason"]:
            findings.append(Finding(cand["path"], "skip", cand["skip_reason"], "", "A"))
            continue
        path = cand["path"]
        if own_issue_re.search(path):
            pat = _match_declared(path, declared_patterns)
            if pat is not None:
                findings.append(
                    Finding(
                        path,
                        "skip",
                        "planned-output-declared",
                        f"own-issue path matches declared output pattern '{pat}' — a plan's "
                        "own declared output is not a carry-over input (#1935)",
                        "A",
                    )
                )
                continue
        findings.append(
            classify(
                {"path": path, "channel": "A"},
                repo_root=repo_root,
                issue=issue,
                check_ref=check_ref,
                local_branch=local_branch,
            )
        )
    a_paths = {c["path"] for c in a_cands}
    scope = plan_issue_scope(plan_text, issue)
    # Compute HF-cited basenames ONCE (#1982): the same plan may cite N bare
    # names, and the regex is content-only (no per-name state).
    hf_cited = extract_hf_cited_basenames(plan_text)
    for name in extract_bare_names(plan_text):
        # No basename pre-filter here: a Channel-A citation of a DIFFERENT
        # file sharing the basename must not suppress this candidate (round-1
        # review concern channel-b-basename-dedup-false-negative). Dedup is by
        # RESOLVED PATH below — only a path already classified via Channel A
        # is skipped, with an explicit ledger row.
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
        # Collect per-name findings into a pending list so the post-classify
        # downgrade (#1982) can rewrite `untracked-local-only` fails to
        # `hf-staged` / `duplicate-resolution` WARNs BEFORE they enter
        # `findings`. The classify() call and every skip branch stay
        # byte-identical to the pre-refactor code path (a1 invariant).
        pending: list[Finding] = []
        foreign: list[str] = []
        for path in resolved:
            if path in a_paths:
                # Same FILE already classified via Channel A — record the
                # dedup so the findings ledger is complete (no silent drop).
                pending.append(
                    Finding(
                        path,
                        "skip",
                        "deduped-channel-a",
                        f"bare name {name} resolves to a path already classified via Channel A",
                        "B",
                    )
                )
                continue
            path_issue = _issue_of_path(path)
            if path_issue is not None and path_issue != issue:
                foreign.append(path)  # summarized per NAME below, never the ladder
                continue
            if path_issue == issue:
                matched: str | None = None
                if name in declared_basenames:
                    matched = f"path-less declared name '{name}'"
                else:
                    pat = _match_declared(path, declared_patterns)
                    if pat is not None:
                        matched = f"declared output pattern '{pat}'"
                if matched is not None:
                    pending.append(
                        Finding(
                            path,
                            "skip",
                            "bare-name-declared-output",
                            f"own-issue resolution of bare name {name} matches {matched} — "
                            "a plan's own declared output is not a carry-over input (#1935)",
                            "B",
                        )
                    )
                    continue
            # Undeclared own-issue (or unparseable) resolution: FULL ladder —
            # the #1434 protection, untouched.
            pending.append(
                classify(
                    {"path": path, "channel": "B"},
                    repo_root=repo_root,
                    issue=issue,
                    check_ref=check_ref,
                    local_branch=local_branch,
                )
            )
        # Post-classify downgrade (#1982): #1979 HF-staged + #1739 in-ref
        # sibling. The helper preserves the #1434 protection (no-op when
        # neither evidence branch fires). Extracted from run_check() to hold
        # its cyclomatic complexity under the C901 threshold.
        _downgrade_untracked_local_only(pending, name, hf_cited)
        findings.extend(pending)
        if foreign:
            head = ", ".join(foreign[:5])
            more = f" (+{len(foreign) - 5} more)" if len(foreign) > 5 else ""
            findings.append(
                Finding(
                    name,
                    "warn",
                    "bare-name-foreign-issue",
                    f"bare-name resolution(s) under FOREIGN issue dirs — not a provable "
                    f"carry-over input ({len(foreign)} path(s): {head}{more}); cite the full "
                    "repo-relative path if this IS a consumed input (#1935)",
                    "B",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Rsync-lane coverage downgrade (#1835 — the SLURM lanes' materialization is an
# rsync of RSYNC_INCLUDE_PATHS, not a git clone, so git-reachability is
# necessary but NOT sufficient there).
# ---------------------------------------------------------------------------


def rsync_cover_set(extra_paths: list[str] | None) -> list[str]:
    """De-dot-anchored RSYNC_INCLUDE_PATHS + `data/` root + --extra-sync-path values.

    Imports the include set from ``explore_persona_space.backends.slurm`` (the
    single source of truth the lane's ``build_rsync_command`` consumes) so the
    gate can never drift from the launch. ``validate_extra_sync_paths`` raises
    ``ValueError`` on a malformed extra path — the caller maps that to exit 2.

    ``RSYNC_DATA_INCLUDE_ROOT`` (#2212): the lane derives its ``data/``
    include entries from the git index at dispatch
    (``tracked_data_include_paths``), but this gate is STATIC (no
    ``src_root``), so it covers the whole ``data/`` root instead — a
    deliberate, documented over-approximation: the gate only ever evaluates
    COMMITTED citations (git-reachability is checked first), and every
    committed ``data/`` path is in the derived set by construction, so the
    approximation cannot produce a false PASS for a citation the lane would
    actually miss.
    """
    from explore_persona_space.backends.slurm import (
        RSYNC_DATA_INCLUDE_ROOT,
        RSYNC_INCLUDE_PATHS,
        validate_extra_sync_paths,
    )

    cover = [p.removeprefix("./") for p in RSYNC_INCLUDE_PATHS]
    cover.append(RSYNC_DATA_INCLUDE_ROOT)
    cover.extend(p.removeprefix("./") for p in validate_extra_sync_paths(extra_paths or ()))
    return list(dict.fromkeys(cover))


def rsync_covered(path: str, cover_set: list[str]) -> bool:
    """True iff `path` is covered by a cover-set prefix (exact or dir-prefix)."""
    return any(path == p or path.startswith(p + "/") for p in cover_set)


def rsync_extra_cover(extra_paths: list[str] | None) -> list[str]:
    """De-dot-anchored ``--extra-sync-path`` values ONLY (no include trees).

    The extra rsync (``build_extra_rsync_command``) is deliberately
    EXCLUDE-FREE, so a path covered by one of these prefixes is genuinely
    staged even when a component matches ``RSYNC_EXCLUDE_PATTERNS`` — the
    downgrade's exclude check (#1915) is suppressed for extra-covered paths.
    ``validate_extra_sync_paths`` raises ``ValueError`` on a malformed
    entry — the caller maps that to exit 2 (same contract as
    ``rsync_cover_set``).
    """
    from explore_persona_space.backends.slurm import validate_extra_sync_paths

    return [p.removeprefix("./") for p in validate_extra_sync_paths(extra_paths or ())]


def rsync_excluded(path: str, exclude_patterns: tuple[str, ...] | None = None) -> str | None:
    """First ``RSYNC_EXCLUDE_PATTERNS`` entry matching `path`, or None (#1915).

    Models the main SLURM-lane rsync's ``--exclude <pat>`` semantics
    (``build_rsync_command`` threads one ``--exclude`` per entry): rsync
    matches slash-free patterns at EVERY path depth, so a citation nested
    under an excluded directory name INSIDE an include tree (e.g.
    ``tests/fixtures/eval_results/a.json`` under the ``./tests`` tree) is
    guaranteed absent on the instance despite include-tree membership.

    Matching rules — conservative, failing toward a cheap false FAIL whose
    remediation (``--extra-sync-path``) structurally works, never a
    stranded false PASS:

    - A slash-free pattern (``__pycache__/``, ``*.pyc``, ``eval_results/``)
      is ``fnmatch.fnmatchcase``'d against EVERY path segment. Dir-only
      (trailing-``/``) patterns are checked against the FINAL segment too —
      a deliberate deviation (rsync applies dir-only patterns to
      directories, but this gate cannot tell a file from a dir), so a file
      literally named like an excluded dir yields a cheap false FAIL.
    - A slash-bearing pattern (``.claude/worktrees/``): rsync matches a
      non-``/``-anchored slash-bearing pattern against the END of the
      pathname and excludes the matched directory during traversal, so the
      check is segment-sequence CONTAINMENT of the de-dotted, de-slashed
      core — NOT a transfer-root prefix match. Unreachable inside today's
      include trees; kept so the semantics stay honest. Wildcards inside a
      slash-bearing core are matched literally (none exist today).

    Lazily imports the constant (like ``rsync_cover_set``) when
    ``exclude_patterns`` is None.
    """
    if exclude_patterns is None:
        from explore_persona_space.backends.slurm import RSYNC_EXCLUDE_PATTERNS

        exclude_patterns = RSYNC_EXCLUDE_PATTERNS
    segs = [s for s in path.split("/") if s]
    for pat in exclude_patterns:
        core = pat.rstrip("/").removeprefix("./")
        if not core:
            continue
        if "/" in core:
            if f"/{path.strip('/')}/".find(f"/{core}/") != -1:
                return pat
        elif any(fnmatch.fnmatchcase(seg, core) for seg in segs):
            return pat
    return None


def apply_rsync_lane_downgrade(
    findings: list[Finding],
    *,
    cover_set: list[str],
    extra_cover: list[str] | tuple[str, ...] = (),
) -> list[Finding]:
    """Post-classification downgrade for rsync-materialized SLURM lanes (#1835).

    A ``Finding(verdict='pass', reason='in-ref')`` downgrades to
    ``fail`` / ``rsync-lane-not-synced`` when EITHER (a) its path is NOT
    covered by RSYNC_INCLUDE_PATHS + the extra-sync paths — the lane's
    scratch tree is an rsync of the include set, so a git-reachable citation
    outside it is guaranteed absent on the instance (#1689: fellows job
    15188 died at first read on a gate-certified committed input) — OR (b)
    it is covered ONLY by a main include tree AND a path component matches
    an ``RSYNC_EXCLUDE_PATTERNS`` entry (#1915: the main rsync excludes at
    every depth, so include-tree membership is necessary but not
    sufficient). A path covered by an ``--extra-sync-path`` prefix
    (`extra_cover`) is NEVER downgraded by (b) — the extra rsync applies no
    excludes. Every other verdict/reason — warns, skips, the clone-lane
    FAILs — is untouched.
    """
    out: list[Finding] = []
    extra_list = list(extra_cover)
    for f in findings:
        if f.verdict != "pass" or f.reason != "in-ref":
            out.append(f)
            continue
        covered = rsync_covered(f.path, cover_set)
        pat: str | None = None
        if covered and not rsync_covered(f.path, extra_list):
            pat = rsync_excluded(f.path)
        if covered and pat is None:
            out.append(f)
            continue
        prefix = f.path.rsplit("/", 1)[0] if "/" in f.path else f.path
        if pat is not None:
            detail = (
                "inside an rsync include tree but a path component matches "
                f"RSYNC_EXCLUDE_PATTERNS entry '{pat}' — the main rsync excludes it at "
                f"every depth; re-dispatch with --extra-sync-path {f.path} (or a "
                f"covering prefix, e.g. --extra-sync-path {prefix}) on BOTH this gate "
                "and dispatch_issue.py launch (the extra rsync applies no excludes)"
            )
        else:
            detail = (
                "git-reachable but NOT in the SLURM lane's rsync set "
                "(RSYNC_INCLUDE_PATHS + extra-sync paths) — re-dispatch with "
                f"--extra-sync-path {f.path} (or a covering prefix, e.g. "
                f"--extra-sync-path {prefix}) on BOTH this gate and "
                "dispatch_issue.py launch"
            )
        out.append(Finding(f.path, "fail", "rsync-lane-not-synced", detail, f.channel))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_repo_root() -> Path | None:
    """Repo root from the cwd's `--git-common-dir` parent (worktree-safe).

    Runs under the same sanitized env as `_git` (#2263 MF-C) so an inherited
    GIT_DIR/GIT_WORK_TREE cannot silently repoint the default repo root.
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
        env=_sanitized_git_env(),
    )
    if proc.returncode != 0:
        return None
    return Path(proc.stdout.strip()).parent


def _repo_branch_arg(value: str) -> str:
    """argparse type for --repo-branch: a bare branch name.

    Rejects empty/whitespace values and `origin/`-prefixed refs (the gate
    probes origin/<branch> itself; a raw ref override is --ref's job).
    """
    if not value.strip():
        raise argparse.ArgumentTypeError("--repo-branch needs a non-empty branch name")
    if value.startswith("origin/"):
        raise argparse.ArgumentTypeError(
            "pass the bare branch name without the `origin/` prefix (the gate probes "
            "origin/<branch> itself); use --ref for a raw ref override"
        )
    return value


def _validate_cli_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Post-parse mode validation (#2263); `parser.error` exits 2.

    Resolver mode (--print-repo-branch) takes only --issue/--repo-root —
    every check-mode flag is refused loud so a fence typo cannot half-run
    the gate; check mode still REQUIRES --plan.
    """
    if args.print_repo_branch:
        forbidden = [
            ("--plan", args.plan is not None),
            ("--ref", args.ref is not None),
            ("--repo-branch", args.repo_branch is not None),
            ("--extra-sync-path", bool(args.extra_sync_path)),
            ("--no-fetch", args.no_fetch),
            ("--json", args.as_json),
            ("--lane", args.lane != "clone"),
        ]
        bad = [flag for flag, hit in forbidden if hit]
        if bad:
            parser.error(
                f"--print-repo-branch combines only with --issue/--repo-root (got {', '.join(bad)})"
            )
    elif args.plan is None:
        parser.error("--plan is required")


def _print_repo_branch_mode(repo_root: Path, issue: int) -> int:
    """Execute --print-repo-branch: the bare branch name on stdout, exit 0.

    On a derivation refusal: ERROR on stderr, EMPTY stdout (load-bearing —
    the fences' ${REPO_BRANCH:?} guard keys on the empty stdout), exit 2.
    """
    try:
        branch = derive_worktree_repo_branch(repo_root, issue)
    except RepoBranchDerivationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(branch)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_carryover_inputs.py",
        description=(
            "Verify plan-cited local repo input files are reachable in the git tree "
            "the compute lane's clone will materialize (/issue Step 6a.5 second stanza)."
        ),
    )
    parser.add_argument(
        "--plan",
        default=None,
        help="path to the approved plan markdown (required except under --print-repo-branch)",
    )
    parser.add_argument("--issue", type=int, required=True, help="task/issue number")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repo root (default: parent of `git rev-parse --git-common-dir`, worktree-safe)",
    )
    parser.add_argument(
        "--print-repo-branch",
        action="store_true",
        help=(
            "print the ONE live issue-<N>-scoped worktree branch and exit — the shared "
            "issue-scoped branch resolver BOTH Step 6 fences call (#2263). Exit 2 with "
            "EMPTY stdout on zero or multiple candidates; combines only with "
            "--issue/--repo-root"
        ),
    )
    ref_group = parser.add_mutually_exclusive_group()
    ref_group.add_argument(
        "--ref",
        default=None,
        help=(
            "check ref override — wins over --repo-branch and worktree inference, and "
            "BYPASSES the ref-existence guard (it can deliberately grade a tree the "
            "dispatch will not materialize, e.g. RunPod BOOTSTRAP_BRANCH defaulting to main)"
        ),
    )
    ref_group.add_argument(
        "--repo-branch",
        type=_repo_branch_arg,
        default=None,
        help=(
            "the dispatch's --repo-branch value, mirrored to this gate (#2263): the check "
            "ref becomes origin/<branch>, REFUSED (exit 2) when absent from origin after "
            "the bounded fetch; `main` is legal (deliberate wholly-main-resident workload). "
            "Pass the SAME value to `dispatch_issue.py launch --repo-branch`"
        ),
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help=(
            "skip the bounded per-branch `git fetch origin ...` calls (tests / corpus "
            "sweeps); resolution then reads possibly-STALE locally-known origin/* refs — "
            "a just-pushed sibling branch is invisible until fetched"
        ),
    )
    parser.add_argument(
        "--lane",
        choices=("clone", "rsync"),
        default="clone",
        help=(
            "materialization lane of the dispatch target (#1835): 'clone' "
            "(GCE / RunPod git clone — the default; byte-identical to the "
            "pre-#1835 behavior) or 'rsync' (the SLURM lanes — "
            "router._PER_CLUSTER_LANES: nibi/fir/mila/fellows, plus the legacy "
            "'cluster' alias — whose scratch tree is an rsync of "
            "RSYNC_INCLUDE_PATHS: an in-ref citation NOT covered by "
            "RSYNC_INCLUDE_PATHS + --extra-sync-path — or covered only by an "
            "include tree while a path component matches an "
            "RSYNC_EXCLUDE_PATTERNS entry (#1915) — downgrades from PASS to "
            "FAIL(rsync-lane-not-synced)). The #1935 declared-output skips "
            "(planned-output-declared / bare-name-declared-output) and the "
            "bare-name-foreign-issue warn are untouched by the downgrade "
            "either lane (it rewrites pass/in-ref rows only)"
        ),
    )
    parser.add_argument(
        "--extra-sync-path",
        action="append",
        default=None,
        metavar="REPO_REL_PATH",
        help=(
            "repo-relative path (repeatable, #1835) the launch will ALSO pass "
            "to `dispatch_issue.py launch --extra-sync-path`; extends the "
            "rsync-lane coverage set (validated either lane; the downgrade "
            "applies only under --lane rsync). Compose this gate call and the "
            "launch from ONE variable so the two sets cannot drift."
        ),
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="JSON findings")
    args = parser.parse_args(argv)
    _validate_cli_mode(parser, args)

    repo_root = args.repo_root if args.repo_root is not None else _default_repo_root()
    if repo_root is None:
        print(
            "ERROR: cannot resolve repo root (not inside a git tree?) — pass --repo-root",
            file=sys.stderr,
        )
        return 2
    repo_root = Path(repo_root).resolve()

    if args.print_repo_branch:
        return _print_repo_branch_mode(repo_root, args.issue)

    plan_path = Path(args.plan)
    try:
        plan_text = plan_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # Fail loud, never exit 0 on an unreadable/undecodable plan (same
        # contract as a missing plan in the first stanza).
        print(f"ERROR: cannot read plan {plan_path}: {exc}", file=sys.stderr)
        return 2

    # #1835: normalize --extra-sync-path either lane (a malformed path is a
    # usage error, exit 2 — same contract as dispatch_issue.py's parse-time
    # guard); the coverage DOWNGRADE applies only under --lane rsync.
    extra_sync_paths: list[str] = []
    if args.extra_sync_path:
        from explore_persona_space.backends.slurm import validate_extra_sync_paths

        try:
            extra_sync_paths = [
                p.removeprefix("./") for p in validate_extra_sync_paths(args.extra_sync_path)
            ]
        except ValueError as exc:
            print(f"ERROR: invalid --extra-sync-path: {exc}", file=sys.stderr)
            return 2

    try:
        if args.ref:
            # Rung 1: an explicit --ref is taken VERBATIM — no resolver call,
            # no existence guard (it can deliberately grade a tree the
            # dispatch will not materialize; #2263).
            resolved = ResolvedRef(args.ref, "ref-flag")
        else:
            resolved = resolve_check_ref(
                repo_root,
                args.issue,
                fetch=not args.no_fetch,
                repo_branch=args.repo_branch,
            )
        check_ref, ref_source = resolved
        local_branch = derive_local_branch(check_ref, ref_source, args.issue)
        findings = run_check(
            plan_text,
            repo_root=repo_root,
            issue=args.issue,
            check_ref=check_ref,
            local_branch=local_branch,
        )
    except CheckRefResolutionError as exc:
        # A refusal is a USAGE-class outcome (exit 2, like a missing plan):
        # distinct from a citation FAIL (exit 1) and never silently skipped.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired as exc:
        # A hung git probe is an infra fault, not a verdict: fail CLOSED with
        # a clean message (exit 1 blocks dispatch; the stanza's re-run path
        # covers the retry) instead of an uncaught traceback.
        print(f"ERROR: git probe timed out ({exc.cmd}); failing closed", file=sys.stderr)
        return 1
    if args.lane == "rsync":
        findings = apply_rsync_lane_downgrade(
            findings,
            cover_set=rsync_cover_set(args.extra_sync_path),
            extra_cover=rsync_extra_cover(args.extra_sync_path),
        )
    n_fail = sum(f.verdict == "fail" for f in findings)
    n_warn = sum(f.verdict == "warn" for f in findings)

    if args.as_json:
        print(
            json.dumps(
                {
                    "plan": str(plan_path),
                    "issue": args.issue,
                    "check_ref": check_ref,
                    "check_ref_source": ref_source,
                    "lane": args.lane,
                    "extra_sync_paths": extra_sync_paths,
                    "n_fail": n_fail,
                    "n_warn": n_warn,
                    "findings": [dataclasses.asdict(f) for f in findings],
                },
                indent=2,
            )
        )
    else:
        for f in findings:
            # ref= on EVERY severity (#2263): a wrong-tree grade must be
            # visible on the row itself, not only in the summary.
            line = f"[{f.verdict.upper():<4}] {f.path} ref={check_ref} reason={f.reason}"
            if f.detail:
                line += f" — {f.detail}"
            print(line)
        print(
            f"checked {len(findings)} citation(s) against {check_ref} "
            f"[ref source: {ref_source}]: {n_fail} fail / {n_warn} warn"
        )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
