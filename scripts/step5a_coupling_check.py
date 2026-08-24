#!/usr/bin/env python3
"""Step 5a coupling / cap-coherence diagnosis (#2327; incidents #2321, #2168).

Read-only ADVISORY detector run right after the Step 5a spec-freshness sync.
It catches "half-sync" states the family-atomic dirty-skip legitimately
produces: a governed doc synced byte-identical to origin/main while the
DERIVED admission state that co-landed with it on main (a size cap in the
lint family, or a co-landed sibling file) was withheld because its family
was dirty on the branch. The symptom downstream is a Step 9c gate red on
bytes identical to origin/main (#2321: 09-step-5.md synced while the
grandfather raise stayed branch-vintage; #2168 INSTANCE 3: one file of a
co-landed sibling pair synced while its pair stayed stale).

Why STATE-based, not commit-grain
---------------------------------
An unrestricted commit-grain detector ("warn whenever a main commit's file
set is split across synced/withheld families") was rejected: a census of
main-history commits found co-landing across families is the NORMAL commit
shape (most workflow-fix commits touch a step doc + a helper + tests), so
commit-grain firing is mostly noise. Two exceptions survive, each anchored
to a demonstrated failure:

* Arm A (cap coherence) keys on the CURRENT STATE of doc bytes vs the two
  cap vintages — it fires only when the branch-effective cap REJECTS bytes
  that origin/main's cap ADMITS (or names the both-reject cell distinctly).
  A skew-aware-cap-check alternative (teach the gate itself to consult
  main's caps) was rejected because it would change what the gate ENFORCES:
  the gate must keep evaluating the branch's own lint against the branch's
  own tree; this helper only pre-diagnoses the mismatch loudly.
* Arm B (sibling split) keeps the commit grain but restricts it to
  ISSUE-KEYED sibling files co-landed by ONE main commit — the #2168 pair
  shape — where "these files land together" is a real invariant.

Three labels, one divergence basis
----------------------------------
divergence_set = tracked worktree-vs-origin/main diff (`git diff
--name-only origin/main`, which compares the origin/main commit against the
WORKING TREE, staged + unstaged) UNION untracked files. A path NOT in
divergence_set has worktree bytes == origin/main bytes by construction.
The same basis drives the Arm A short-circuit, the per-doc discriminator,
and Arm B's fresh/stale split — never a mix of HEAD-based and tree-based
reads.

| label           | predicate                                               |
|-----------------|---------------------------------------------------------|
| cap-skew        | doc bytes == origin/main (not in divergence_set)        |
|                 | AND branch-effective cap REJECTS the doc's size         |
|                 | AND origin/main-effective cap ADMITS it                 |
| cap-red-on-main | doc bytes == origin/main AND BOTH caps reject           |
|                 | (main itself is red — merging cannot fix it)            |
| sibling-split   | ONE first-parent main commit in MB..origin/main touched |
|                 | >=2 issue-keyed sibling files of the SAME issue M       |
|                 | (M != own issue), and the branch now holds >=1 of them  |
|                 | == origin/main (fresh) and >=1 differing (stale)        |

rc semantics (advisory / fail-open contract): 0 = verdicts delivered (with
or without WARNs); nonzero = the helper itself was undecidable — the caller
prints a loud unavailable-line to stderr and CONTINUES (never blocks the
round). Caps are extracted from workflow_lint.py TEXT via ast (module-level
Assign AND AnnAssign) — never by importing it (its import runs
_load_agent_spec_caps() against the live tree, and import would execute
branch code while diagnosing main's).
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Parity-pinned against scripts/step5a_sibling_probe.py (full-path search,
# never basename-only: src/.../experiments/issue_<N>/... has no basename key).
_ISSUE_NUMBER_RE = re.compile(r"issue_?(\d+)")

# Parity-pinned against the Step 5a sibling arm's diff pathspecs
# (.claude/skills/issue/steps/09-step-5.md, the `git diff --name-only
# origin/main -- ':(glob)...'` line).
SIBLING_PATHSPECS = (
    ":(glob)scripts/issue[0-9]*_*.py",
    ":(glob)scripts/issue[0-9]*_*.sh",
    ":(glob)tests/test_issue[0-9]*_*.py",
    ":(glob)src/explore_persona_space/experiments/issue[0-9]*/**",
    ":(glob)src/explore_persona_space/experiments/issue_[0-9]*/**",
)

# The two files that carry derived admission state (Arm A's engagement key).
CAP_SOURCE_PATHS = (
    "scripts/workflow_lint.py",
    ".claude/config/agent_spec_size_caps.txt",
)

_AGENT_CAPS_REL = ".claude/config/agent_spec_size_caps.txt"
_LINT_REL = "scripts/workflow_lint.py"

_CAP_NAMES = (
    "SKILL_DOC_SIZE_GRANDFATHER",
    "_LESSONS_MAX_BYTES",
    "SKILL_DOC_FAIL_BYTES",
    "AGENT_SPEC_FAIL_BYTES",
    "SKILL_DOC_EXEMPT_DIR_SEGMENTS",
    "SKILL_DOC_GENERATED_EXEMPT",
)


def _notice(msg: str) -> None:
    print(f"[step5a] coupling check notice: {msg}")


def _git(wt: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(wt), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args[:3])}... rc={proc.returncode}: {proc.stderr.strip()[:300]}"
        )
    return proc.stdout


def _git_names(wt: Path, *args: str) -> set[str]:
    return {ln.strip() for ln in _git(wt, *args).splitlines() if ln.strip()}


def _git_show(wt: Path, ref_path: str) -> str | None:
    """Content of `ref:path`, or None when the path is absent at the ref."""
    proc = subprocess.run(["git", "-C", str(wt), "show", ref_path], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout


def _literal_value(node: ast.expr):
    """literal_eval extended to the frozenset({...}) Call form the lint uses."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "set", "tuple", "list"}
        and not node.keywords
    ):
        if not node.args:
            return frozenset()
        if len(node.args) == 1:
            return frozenset(ast.literal_eval(node.args[0]))
        raise ValueError("multi-arg container call")
    return ast.literal_eval(node)


@dataclass
class Caps:
    """Cap constants extracted from one vintage of workflow_lint.py."""

    values: dict[str, object]
    missing: tuple[str, ...]


def _extract_caps(py_src: str) -> Caps:
    """Extract the six cap names from module-level Assign AND AnnAssign nodes.

    #2303's raise landed as an AnnAssign (`SKILL_DOC_SIZE_GRANDFATHER:
    dict[str, int] = {...}`); an Assign-only walker reads it as missing and
    silently skips the regime — hence BOTH node classes are accepted, and a
    value-less annotation (`name: int`) is skipped, never a crash.
    """
    tree = ast.parse(py_src)
    found: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            names = [node.target.id]
            value = node.value
        else:
            continue
        for name in names:
            if name in _CAP_NAMES and name not in found:
                try:
                    found[name] = _literal_value(value)
                except (ValueError, SyntaxError, TypeError):
                    pass  # non-literal binding -> treated as missing (advisory)
    # Type sanity: a wrong-shaped literal degrades to missing, never a crash.
    gf = found.get("SKILL_DOC_SIZE_GRANDFATHER")
    if gf is not None and not (
        isinstance(gf, dict)
        and all(isinstance(k, str) and isinstance(v, int) for k, v in gf.items())
    ):
        del found["SKILL_DOC_SIZE_GRANDFATHER"]
    for int_name in ("_LESSONS_MAX_BYTES", "SKILL_DOC_FAIL_BYTES", "AGENT_SPEC_FAIL_BYTES"):
        if int_name in found and not isinstance(found[int_name], int):
            del found[int_name]
    for set_name in ("SKILL_DOC_EXEMPT_DIR_SEGMENTS", "SKILL_DOC_GENERATED_EXEMPT"):
        if set_name in found:
            try:
                found[set_name] = frozenset(found[set_name])  # type: ignore[arg-type]
            except TypeError:
                del found[set_name]
    missing = tuple(n for n in _CAP_NAMES if n not in found)
    return Caps(values=found, missing=missing)


def _parse_agent_caps(text: str) -> dict[str, int]:
    """Mirror of workflow_lint._load_agent_spec_caps line grammar.

    `<agent-file-name> <cap-bytes>` per line; `#` starts a comment;
    underscores allowed in the int; duplicate names raise; malformed lines
    raise ValueError (the caller degrades advisorily).
    """
    caps: dict[str, int] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"agent caps line {lineno}: expected '<name> <cap>'")
        name, cap_str = parts
        cap = int(cap_str.replace("_", ""))
        if name in caps:
            raise ValueError(f"agent caps line {lineno}: duplicate entry {name!r}")
        caps[name] = cap
    return caps


@dataclass
class _Side:
    """One vintage (branch worktree, or origin/main) of the cap surface."""

    caps: Caps
    agent_caps: dict[str, int] | None  # None => agents regime undecidable


def _load_side_branch(wt: Path) -> _Side | None:
    lint_path = wt / _LINT_REL
    if not lint_path.is_file():
        _notice(f"{_LINT_REL} absent in worktree — cap-coherence arm skipped")
        return None
    try:
        caps = _extract_caps(lint_path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        _notice(f"worktree {_LINT_REL} unparseable ({exc}) — cap-coherence arm skipped")
        return None
    agent_caps: dict[str, int] | None
    caps_path = wt / _AGENT_CAPS_REL
    if not caps_path.is_file():
        _notice(f"{_AGENT_CAPS_REL} absent in worktree — agents fall back to AGENT_SPEC_FAIL_BYTES")
        agent_caps = {}
    else:
        try:
            agent_caps = _parse_agent_caps(caps_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            _notice(f"worktree {_AGENT_CAPS_REL} malformed ({exc}) — agents regime skipped")
            agent_caps = None
    return _Side(caps=caps, agent_caps=agent_caps)


def _load_side_main(wt: Path) -> _Side | None:
    lint_src = _git_show(wt, f"origin/main:{_LINT_REL}")
    if lint_src is None:
        _notice(f"{_LINT_REL} absent at origin/main — cap-coherence arm skipped")
        return None
    try:
        caps = _extract_caps(lint_src)
    except SyntaxError as exc:
        _notice(f"origin/main {_LINT_REL} unparseable ({exc}) — cap-coherence arm skipped")
        return None
    agent_caps: dict[str, int] | None
    caps_text = _git_show(wt, f"origin/main:{_AGENT_CAPS_REL}")
    if caps_text is None:
        _notice(
            f"{_AGENT_CAPS_REL} absent at origin/main — agents fall back to AGENT_SPEC_FAIL_BYTES"
        )
        agent_caps = {}
    else:
        try:
            agent_caps = _parse_agent_caps(caps_text)
        except ValueError as exc:
            _notice(f"origin/main {_AGENT_CAPS_REL} malformed ({exc}) — agents regime skipped")
            agent_caps = None
    return _Side(caps=caps, agent_caps=agent_caps)


def _skills_regime_ready(side: _Side) -> bool:
    return all(
        n in side.caps.values
        for n in (
            "SKILL_DOC_SIZE_GRANDFATHER",
            "SKILL_DOC_FAIL_BYTES",
            "SKILL_DOC_EXEMPT_DIR_SEGMENTS",
            "SKILL_DOC_GENERATED_EXEMPT",
        )
    )


def _skills_cap(side: _Side, rel_sk: str) -> tuple[int | None, str]:
    """Effective skills-doc cap under one side; (None, reason) => admits (exempt)."""
    segs = side.caps.values["SKILL_DOC_EXEMPT_DIR_SEGMENTS"]
    gen = side.caps.values["SKILL_DOC_GENERATED_EXEMPT"]
    if rel_sk in gen or any(seg in segs for seg in rel_sk.split("/")[:-1]):  # type: ignore[operator]
        return None, "exempt"
    gf = side.caps.values["SKILL_DOC_SIZE_GRANDFATHER"]
    if rel_sk in gf:  # type: ignore[operator]
        return gf[rel_sk], f"SKILL_DOC_SIZE_GRANDFATHER[{rel_sk!r}]"  # type: ignore[index]
    return side.caps.values["SKILL_DOC_FAIL_BYTES"], "SKILL_DOC_FAIL_BYTES"  # type: ignore[return-value]


def _agents_cap(side: _Side, name: str) -> tuple[int, str]:
    assert side.agent_caps is not None
    if name in side.agent_caps:
        return side.agent_caps[name], f"agent_spec_size_caps.txt[{name!r}]"
    return side.caps.values["AGENT_SPEC_FAIL_BYTES"], "AGENT_SPEC_FAIL_BYTES"  # type: ignore[return-value]


def check_cap_coherence(
    wt: Path, divergence_set: set[str], merge_base: str
) -> list[tuple[str, str]]:
    """Arm A: cap-skew / cap-red-on-main over the three size-cap regimes."""
    if not any(p in divergence_set for p in CAP_SOURCE_PATHS):
        return []  # cap surface == origin/main => no vintage mismatch possible
    branch = _load_side_branch(wt)
    main = _load_side_main(wt)
    if branch is None or main is None:
        return []

    skills_ready = _skills_regime_ready(branch) and _skills_regime_ready(main)
    if not skills_ready:
        miss = sorted(set(branch.caps.missing) | set(main.caps.missing))
        _notice(f"skills regime skipped — constants missing/non-literal: {miss}")
    lessons_ready = (
        "_LESSONS_MAX_BYTES" in branch.caps.values and "_LESSONS_MAX_BYTES" in main.caps.values
    )
    if not lessons_ready:
        _notice("LESSONS regime skipped — _LESSONS_MAX_BYTES missing/non-literal on a side")
    agents_ready = (
        "AGENT_SPEC_FAIL_BYTES" in branch.caps.values
        and "AGENT_SPEC_FAIL_BYTES" in main.caps.values
        and branch.agent_caps is not None
        and main.agent_caps is not None
    )
    if not agents_ready:
        _notice("agents regime skipped — AGENT_SPEC_FAIL_BYTES or caps file undecidable on a side")

    # (rel_path, regime, regime_key) for every governed doc present in the tree.
    docs: list[tuple[str, str, str]] = []
    skills_root = wt / ".claude" / "skills"
    if skills_ready and skills_root.is_dir():
        for p in sorted(skills_root.rglob("*.md")):
            rel = p.relative_to(wt).as_posix()
            rel_sk = p.relative_to(skills_root).as_posix()
            docs.append((rel, "skills", rel_sk))
    agents_root = wt / ".claude" / "agents"
    if agents_ready and agents_root.is_dir():
        for p in sorted(agents_root.glob("*.md")):
            docs.append((p.relative_to(wt).as_posix(), "agents", p.name))
    lessons = wt / ".claude" / "rules" / "LESSONS.md"
    if lessons_ready and lessons.is_file():
        docs.append((lessons.relative_to(wt).as_posix(), "lessons", ""))

    warns: list[tuple[str, str]] = []
    for rel, regime, key in docs:
        if rel in divergence_set:
            continue  # branch-authored content: the gate red (if any) is real
        path = wt / rel
        if not path.is_file():
            continue
        size = path.stat().st_size
        if regime == "skills":
            b_cap, b_desc = _skills_cap(branch, key)
            m_cap, _ = _skills_cap(main, key)
        elif regime == "agents":
            b_cap, b_desc = _agents_cap(branch, key)
            m_cap, _ = _agents_cap(main, key)
        else:
            b_cap, b_desc = branch.caps.values["_LESSONS_MAX_BYTES"], "_LESSONS_MAX_BYTES"  # type: ignore[assignment]
            m_cap, _ = main.caps.values["_LESSONS_MAX_BYTES"], ""  # type: ignore[assignment]
        if b_cap is None or size <= b_cap:
            continue  # branch admits => the gate will not red on this doc
        if m_cap is None or size <= m_cap:
            main_desc = "exempt" if m_cap is None else str(m_cap)
            warns.append(
                (
                    "cap-skew",
                    f"{rel} = {size} B exceeds the branch's {b_desc} = {b_cap} while "
                    f"origin/main's cap ({main_desc}) admits these bytes — consistent with "
                    f"the Step 5a sync copying the doc while the dirty lint family withheld "
                    f"its derived cap (#2321). If the branch deliberately lowered this cap, "
                    f"this is that choice surfacing: check "
                    f"`git -C {wt} log {merge_base}..HEAD -- {_LINT_REL}`. The Step 9c size "
                    f"pins will red on bytes identical to origin/main. Remedy: bring the "
                    f"branch current — `git -C {wt} merge origin/main` (#2311/#2168) — or, "
                    f"if the branch never touched this cap, restore it byte-exact from "
                    f"origin/main.",
                )
            )
        else:
            warns.append(
                (
                    "cap-red-on-main",
                    f"{rel} = {size} B exceeds BOTH the branch cap ({b_desc} = {b_cap}) "
                    f"and origin/main's cap ({m_cap}) — origin/main itself rejects these "
                    f"bytes, so this is NOT a half-sync: merging origin/main will not fix "
                    f"it (expect a main-side re-ratchet or an upstream fix).",
                )
            )
    return warns


def check_sibling_split(
    wt: Path, merge_base: str, own_issue: str | None, divergence_set: set[str]
) -> list[tuple[str, str]]:
    """Arm B: one main commit co-landed >=2 sibling files of issue M; the
    branch now holds mixed vintages of them."""
    out = _git(
        wt,
        "log",
        "--format=%H",
        "--name-only",
        "--diff-merges=first-parent",
        f"{merge_base}..origin/main",
        "--",
        *SIBLING_PATHSPECS,
    )
    current: str | None = None
    bundled: dict[str, list[str]] = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"[0-9a-f]{40}", line):
            current = line
            bundled[current] = []
        elif current is not None:
            bundled[current].append(line)

    # issue M -> {"shas": [...], "fresh": set, "stale": set}
    qualifying: dict[str, dict[str, object]] = {}
    for sha, files in bundled.items():
        by_issue: dict[str, set[str]] = {}
        for f in files:
            m = _ISSUE_NUMBER_RE.search(f)
            if not m:
                continue
            by_issue.setdefault(m.group(1), set()).add(f)
        for issue_m, group in by_issue.items():
            if own_issue is not None and issue_m == own_issue:
                continue  # the branch's own files are its deliverables, not a split
            if len(group) < 2:
                continue
            fresh = {f for f in group if f not in divergence_set and (wt / f).exists()}
            stale = group - fresh
            if fresh and stale:
                entry = qualifying.setdefault(issue_m, {"shas": [], "fresh": set(), "stale": set()})
                entry["shas"].append(sha[:12])  # type: ignore[union-attr]
                entry["fresh"] |= fresh  # type: ignore[operator]
                entry["stale"] |= stale  # type: ignore[operator]

    warns: list[tuple[str, str]] = []
    for issue_m in sorted(qualifying, key=int):
        entry = qualifying[issue_m]
        shas = ", ".join(entry["shas"])  # type: ignore[arg-type]
        fresh_s = ", ".join(sorted(entry["fresh"]))  # type: ignore[arg-type]
        stale_s = ", ".join(sorted(entry["stale"]))  # type: ignore[arg-type]
        warns.append(
            (
                "sibling-split",
                f"issue {issue_m} — main commit(s) {shas} co-landed sibling files; the "
                f"branch now holds {fresh_s} == origin/main while {stale_s} differs from "
                f"origin/main (withheld by the sync or branch-edited) — mixed vintages of "
                f"a co-landed set (#2168 INSTANCE 3). Remedy: `git -C {wt} merge "
                f"origin/main` (#2311), or pair-revert per the Step 5a manual recovery.",
            )
        )
    return warns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worktree", required=True, help="issue worktree root")
    ap.add_argument("--merge-base", required=True, help="merge-base HEAD origin/main")
    ap.add_argument(
        "--own-issue",
        default=None,
        help="this branch's issue number (its own sibling files are never a split)",
    )
    args = ap.parse_args(argv)
    wt = Path(args.worktree)
    try:
        if not wt.is_dir():
            raise RuntimeError(f"worktree {wt} is not a directory")
        divergence_set = _git_names(wt, "diff", "--name-only", "origin/main") | _git_names(
            wt, "ls-files", "--others", "--exclude-standard"
        )
        warns = check_cap_coherence(wt, divergence_set, args.merge_base)
        warns += check_sibling_split(wt, args.merge_base, args.own_issue, divergence_set)
    except Exception as exc:  # undecidable => nonzero; caller degrades advisorily
        print(f"[step5a] coupling check: helper undecidable: {exc}", file=sys.stderr)
        return 1
    for label, msg in warns:
        print(f"[step5a] WARN: {label}: {msg}")
    if warns:
        counts = {
            k: sum(1 for lbl, _ in warns if lbl == k)
            for k in ("cap-skew", "sibling-split", "cap-red-on-main")
        }
        print(
            f"[step5a] coupling check: {counts['cap-skew']} cap-skew, "
            f"{counts['sibling-split']} sibling-split, "
            f"{counts['cap-red-on-main']} cap-red-on-main warning(s)"
        )
    else:
        print("[step5a] coupling check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
