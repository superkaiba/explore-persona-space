"""Tests for ``workflow_lint.check_codex_composer_memory_commit_lens`` (#2473).

The FAIL surface pin (``--check-codex-composer-memory-commit``, bundled into
the no-flags default run): the codex-composer shared contract
(``.claude/rules/codex-composer-common.md``) must keep its
"Your own agent-memory writes" section — the same-turn explicit-path commit
duty (lesson + ``MEMORY.md`` index row in ONE commit), the counter-argument
to the invented mid-round-contamination heuristic, the
``guard_root_code_commit.sh`` literal-pathspec rationale, and the
do-not-defer-to-a-post-merge-sweep literal — plus the Compose-only
Bash-allowlist cross-reference that licenses the commit.

Incident #2473 (observed on #2263 review rounds 4/5/6): three composer
spawns each left a memory lesson uncommitted "to keep it out of the diff
under review" — the #2015 stash-race dominant standing-armer class — and
the orchestrator hand-committed all three.

1.  ``test_lens_passes_on_complete_corpus`` — heading + all four tokens +
    the pre-heading cross-reference present.
2.  ``test_lens_fails_per_missing_token`` — parametrized drops (missing
    file, missing heading, each of the four tokens, missing cross-ref, and
    a token MIGRATED below the next ``## `` heading — the plan §15 item-2
    tightening vs the #2326 heading-to-EOF precedent), each yielding
    EXACTLY ONE deterministic error naming the dropped literal.
3.  ``test_lens_passes_on_live_tree`` — binds the landed #2473 D1 edits
    (the Durability pin).
4.  ``test_check_codex_composer_memory_commit_bundled_in_no_flags`` — the
    two-part behavioral bundling pin (the #1701 / #2326 precedent shape):
    Part A scoped-flag subprocess against a drifted tmp corpus; Part B
    source pin on the ``no_flags`` OR-chain + dispatch ladder (the
    #1385/#1648 silent-unbundling shape).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from workflow_lint import check_codex_composer_memory_commit_lens  # noqa: E402


def _rule_text(
    *,
    heading: bool = True,
    crossref: bool = True,
    same_turn: bool = True,
    memory_md: bool = True,
    guard: bool = True,
    sweep: bool = True,
    migrate_sweep: bool = False,
) -> str:
    """Compose a minimal rule-file corpus; each kwarg drops ONE pinned literal.

    ``migrate_sweep`` keeps the sweep literal in the FILE but moves it BELOW
    the next ``## `` heading, so a heading-to-EOF region would still PASS —
    the bounded region must FAIL it.
    """
    xref = " (§ Your own agent-memory writes)" if crossref else ""
    parts = [
        "# Codex composer common contract (all codex-* twins)\n\n",
        "## Compose-only — NEVER dispatch Codex yourself\n\n",
        "- The only Bash you may run: writing the prompt file, local\n"
        f"  prompt-file validation, and the guarded commit of your own\n"
        f"  agent-memory writes{xref}.\n\n",
        "## Return contract\n\n",
        "Return the prompt-file path as your final text.\n\n",
    ]
    if heading:
        parts.append("## Your own agent-memory writes\n\n")
        line1 = "A memory lesson you save is a tracked write like any other: commit it\n"
        if same_turn:
            line1 += "by explicit path in the SAME turn you write it"
        else:
            line1 += "by explicit path immediately"
        if memory_md:
            line1 += ", together with its\nMEMORY.md index row — ONE commit.\n"
        else:
            line1 += ", together with its\nindex row — ONE commit.\n"
        parts.append(line1)
        if sweep and not migrate_sweep:
            parts.append("Do NOT defer it to a post-merge sweep.\n")
        else:
            parts.append("Do NOT defer it.\n")
        if guard:
            parts.append("Literal paths are load-bearing for guard_root_code_commit.sh.\n")
        parts.append("\n")
    parts.append("## Trailing section\n\ntrailing prose bounding the region.\n")
    if migrate_sweep:
        parts.append("A post-merge sweep is mentioned only down here.\n")
    return "".join(parts)


_DROP_VARIANTS: dict[str, str] = {
    "heading": _rule_text(heading=False),
    "crossref": _rule_text(crossref=False),
    "same-turn": _rule_text(same_turn=False),
    "memory-md": _rule_text(memory_md=False),
    "guard": _rule_text(guard=False),
    "sweep": _rule_text(sweep=False),
    "sweep-migrated": _rule_text(migrate_sweep=True),
}


def _write_corpus(root: Path, drop: str | None = None) -> None:
    """Write the single-surface corpus; ``drop`` names one defect."""
    rule = root / ".claude" / "rules" / "codex-composer-common.md"
    if drop == "file":
        return
    rule.parent.mkdir(parents=True, exist_ok=True)
    rule.write_text(_DROP_VARIANTS.get(drop or "", _rule_text()), encoding="utf-8")


def test_lens_passes_on_complete_corpus(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    assert check_codex_composer_memory_commit_lens(repo_root=tmp_path) == []


@pytest.mark.parametrize(
    ("drop", "token_fragment"),
    [
        ("file", "missing"),
        ("heading", "## Your own agent-memory writes"),
        ("crossref", "§ Your own agent-memory writes"),
        ("same-turn", "in the SAME turn"),
        ("memory-md", "MEMORY.md"),
        ("guard", "guard_root_code_commit.sh"),
        ("sweep", "post-merge sweep"),
        # Region tightening (#2473 plan §15 item 2): a pinned literal that
        # MIGRATES below the next `## ` heading no longer satisfies the scan
        # (the #2326 precedent's heading-to-EOF region would have passed it).
        ("sweep-migrated", "post-merge sweep"),
    ],
)
def test_lens_fails_per_missing_token(tmp_path: Path, drop: str, token_fragment: str) -> None:
    _write_corpus(tmp_path, drop=drop)
    errors = check_codex_composer_memory_commit_lens(repo_root=tmp_path)
    assert len(errors) == 1, f"drop={drop!r} must yield exactly one error; got: {errors}"
    assert "codex-composer-common.md" in errors[0]
    assert token_fragment in errors[0]


def test_lens_passes_on_live_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binds the landed #2473 D1 edits; the standing regression guard for
    future refactors of the composer contract (the Durability pin)."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    errors = check_codex_composer_memory_commit_lens(repo_root=None)
    assert errors == [], f"live tree should carry the #2473 section + cross-ref; got: {errors}"


def test_check_codex_composer_memory_commit_bundled_in_no_flags(tmp_path: Path) -> None:
    """Two-part behavioral bundling pin (the #1701 / #2326 precedent shape).

    Part A — scoped-flag subprocess against a DRIFTED corpus (the section
    dropped), rooted via ``EPS_WORKFLOW_LINT_REPO_ROOT``: proves the flag
    exists, the dispatch calls the function, and it emits its uniquely
    tagged error (nonzero exit).

    Part B — no-flags OR-chain + dispatch-ladder evidence: ``main()``'s
    source names ``args.check_codex_composer_memory_commit`` in BOTH the
    ``no_flags = not (...)`` OR-chain and the ``or no_flags`` dispatch
    ladder — the #1385 / #1648 silent-unbundling shape stays pinned across
    a later dispatch refactor.
    """
    # Part A — scoped-flag subprocess against a drifted corpus.
    _write_corpus(tmp_path, drop="heading")
    workflow_yaml_src = _REPO_ROOT / ".claude" / "workflow.yaml"
    workflow_yaml_dst = tmp_path / ".claude" / "workflow.yaml"
    workflow_yaml_dst.parent.mkdir(parents=True, exist_ok=True)
    workflow_yaml_dst.write_bytes(workflow_yaml_src.read_bytes())
    lint_script = _REPO_ROOT / "scripts" / "workflow_lint.py"
    env = {**os.environ, "EPS_WORKFLOW_LINT_REPO_ROOT": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            str(lint_script),
            "--check-codex-composer-memory-commit",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "## Your own agent-memory writes" in combined, (
        "the #2473 error token is missing from the output — the CLI flag "
        "does not dispatch the check. "
        f"exit={result.returncode}, combined output:\n{combined}"
    )
    assert result.returncode != 0, (
        f"expected nonzero exit under drifted corpus; got exit="
        f"{result.returncode}, combined output:\n{combined}"
    )

    # Part B — OR-chain + dispatch ladder evidence.
    lint_src = lint_script.read_text(encoding="utf-8")
    main_start = lint_src.find("def main(")
    assert main_start >= 0, "could not locate def main( in workflow_lint.py"
    main_end = lint_src.find('if __name__ == "__main__":', main_start)
    assert main_end > main_start, "could not locate main() end sentinel"
    main_src = lint_src[main_start:main_end]
    or_chain_start = main_src.find("no_flags = not (")
    assert or_chain_start >= 0, "no_flags OR-chain not found in main()"
    or_chain_end = main_src.find(")", or_chain_start)
    or_chain_src = main_src[or_chain_start:or_chain_end]
    assert "args.check_codex_composer_memory_commit" in or_chain_src, (
        "args.check_codex_composer_memory_commit is NOT in the no_flags "
        "OR-chain — a bare workflow_lint.py invocation will not fire this "
        f"check. OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_codex_composer_memory_commit or no_flags" in main_src, (
        "args.check_codex_composer_memory_commit is NOT dispatched under "
        "`or no_flags` — the flag is defined but not bundled into the "
        "no-flags default run."
    )
