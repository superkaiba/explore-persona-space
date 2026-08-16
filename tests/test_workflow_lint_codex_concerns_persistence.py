"""Tests for ``workflow_lint.check_codex_concerns_persistence_lens`` (#2326).

The FAIL surface pin (``--check-codex-concerns-persistence``, bundled into
the no-flags default run): the Codex concerns-persistence contract must stay
present across its FOUR surfaces — the issue/SKILL.md poster-duty subsection
(forwarder invocation + resume-recovery clause), the ``CONCERN:: `` row
grammar in both emitting composers' verdict templates, and the
``**Prior-concerns ledger:**`` visibility line in code-reviewer.md Step 0.8.

Incident #2321: a Codex verdict carried 8 "Concerns to persist" items, zero
were persisted, and the round-2 prior-concerns gate walked an empty ledger.

1.  ``test_lens_passes_on_complete_corpus`` — all four surfaces present.
2.  ``test_lens_fails_per_missing_surface`` — 12 parametrized drops, each
    naming the file + missing token.
3.  ``test_lens_passes_on_live_tree`` — binds the landed #2326 edits.
4.  ``test_check_codex_concerns_persistence_bundled_in_no_flags`` — the
    two-part behavioral bundling pin (the #1701 test's precedent shape,
    mirroring tests/test_workflow_lint_smoke_blind_spots.py).
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

from workflow_lint import check_codex_concerns_persistence_lens  # noqa: E402

_SKILL_OK = """\
prose above the recipe.

**Codex concerns persistence at verdict collection (fires at EVERY
marker-mode Codex verdict collection; #2326).** The Codex twins never mutate
`concerns.jsonl`; the orchestrator forwards their machine-readable rows.

    uv run python scripts/persist_verdict_concerns.py <N> --file "$MB" \\
      --by <codex-role> --round <n> --require-block

**Resume recovery (crash between marker post and persist — #2326).**
At ANY resume/decision point where a current-round marker ALREADY EXISTS,
run recovery FIRST, then the row's action.

    uv run python scripts/persist_verdict_concerns.py <N> --file "$MB" \\
      --by <codex-role> --round <n> --require-block

**5c. Apply ensemble decision rule.**

decision table here.
"""

# Heading present, forwarder literal present, recovery clause ABSENT.
_SKILL_NO_RECOVERY = """\
**Codex concerns persistence at verdict collection (#2326).** prose.

    uv run python scripts/persist_verdict_concerns.py <N> --file "$MB"

**5c. Apply ensemble decision rule.**
"""

# Heading present, forwarder literal ABSENT (recovery clause present).
_SKILL_NO_FORWARDER = """\
**Codex concerns persistence at verdict collection (#2326).** prose that
names no script at all.

**Resume recovery (crash between marker post and persist — #2326).** prose.

**5c. Apply ensemble decision rule.**
"""

_CODEX_CR_OK = """\
composer prose.

<!-- epm:code-review-codex v{{revision_round}} -->
**Verdict:** [PASS | FAIL]

## Concerns to persist

CONCERN:: <BLOCKER|CONCERN|NIT> <kebab-case-id> <one-line summary>

<!-- /epm:code-review-codex -->

trailing prose (Marker start tag: <!-- epm:code-review-codex ... -->).
"""

_CODEX_CR_NO_TOKEN = """\
composer prose.

<!-- epm:code-review-codex v{{revision_round}} -->
**Verdict:** [PASS | FAIL]

## Concerns to persist

- prose bullets only, no machine rows

<!-- /epm:code-review-codex -->
"""

_CODEX_CRC_OK = """\
composer prose.

<!-- epm:clean-result-critique-codex v{{revision_round}} -->
**Verdict:** [PASS | REVISE]

### Concerns to persist

CONCERN:: <BLOCKER|CONCERN|NIT> <kebab-case-id> <one-line summary>

<!-- /epm:clean-result-critique-codex -->
"""

_CODEX_CRC_NO_TOKEN = """\
composer prose.

<!-- epm:clean-result-critique-codex v{{revision_round}} -->
**Verdict:** [PASS | REVISE]

<!-- /epm:clean-result-critique-codex -->
"""

_REVIEWER_OK = """\
### Step 0.7: something else

prose.

### Step 0.8: Read prior open binding concerns

Fetch the ledger. Record the ledger state as one verdict-body line —
`**Prior-concerns ledger:** <K open: id1, id2, …>` or
`**Prior-concerns ledger:** empty` — so a vacuous walk is visible (#2326).

### Step 0.9: Git-provenance self-check

prose.
"""

_REVIEWER_NO_LEDGER_LINE = """\
### Step 0.8: Read prior open binding concerns

Fetch the ledger; walk the concerns. No visibility line here.

### Step 0.9: Git-provenance self-check

prose.
"""


def _write_lens_corpus(root: Path, drop: str | None = None) -> None:
    """Write a minimal four-surface corpus; ``drop`` names one defect."""
    skill = root / ".claude" / "skills" / "issue" / "SKILL.md"
    codex_cr = root / ".claude" / "agents" / "codex-code-reviewer.md"
    codex_crc = root / ".claude" / "agents" / "codex-clean-result-critic.md"
    reviewer = root / ".claude" / "agents" / "code-reviewer.md"
    for path in (skill, codex_cr, codex_crc, reviewer):
        path.parent.mkdir(parents=True, exist_ok=True)

    skill_text = _SKILL_OK
    if drop == "skill-heading":
        skill_text = "no subsection here.\n\n**5c. Apply ensemble decision rule.**\n"
    elif drop == "skill-forwarder":
        skill_text = _SKILL_NO_FORWARDER
    elif drop == "skill-recovery":
        skill_text = _SKILL_NO_RECOVERY
    if drop != "skill-file":
        skill.write_text(skill_text)

    cr_text = _CODEX_CR_OK
    if drop == "codex-cr-start-tag":
        # Tag only mid-line (prose mention), never at line start.
        cr_text = "prose Marker start tag: <!-- epm:code-review-codex v1 --> only.\n"
    elif drop == "codex-cr-token":
        cr_text = _CODEX_CR_NO_TOKEN
    if drop != "codex-cr-file":
        codex_cr.write_text(cr_text)

    crc_text = _CODEX_CRC_OK
    if drop == "codex-crc-start-tag":
        crc_text = "prose <!-- epm:clean-result-critique-codex v1 --> mid-line only.\n"
    elif drop == "codex-crc-token":
        crc_text = _CODEX_CRC_NO_TOKEN
    if drop != "codex-crc-file":
        codex_crc.write_text(crc_text)

    reviewer_text = _REVIEWER_OK
    if drop == "reviewer-step08":
        reviewer_text = "### Step 0.9: Git-provenance self-check\n\nprose.\n"
    elif drop == "reviewer-ledger-line":
        reviewer_text = _REVIEWER_NO_LEDGER_LINE
    if drop != "reviewer-file":
        reviewer.write_text(reviewer_text)


def test_lens_passes_on_complete_corpus(tmp_path: Path) -> None:
    _write_lens_corpus(tmp_path)
    assert check_codex_concerns_persistence_lens(repo_root=tmp_path) == []


@pytest.mark.parametrize(
    ("drop", "path_fragment", "token_fragment"),
    [
        ("skill-file", "SKILL.md", "missing"),
        ("skill-heading", "SKILL.md", "Codex concerns persistence at verdict collection"),
        ("skill-forwarder", "SKILL.md", "persist_verdict_concerns.py"),
        ("skill-recovery", "SKILL.md", "Resume recovery"),
        ("codex-cr-file", "codex-code-reviewer.md", "missing"),
        ("codex-cr-start-tag", "codex-code-reviewer.md", "start tag"),
        ("codex-cr-token", "codex-code-reviewer.md", "CONCERN:: "),
        ("codex-crc-file", "codex-clean-result-critic.md", "missing"),
        ("codex-crc-start-tag", "codex-clean-result-critic.md", "start tag"),
        ("codex-crc-token", "codex-clean-result-critic.md", "CONCERN:: "),
        ("reviewer-file", "code-reviewer.md", "missing"),
        ("reviewer-step08", "code-reviewer.md", "Step 0.8"),
    ],
)
def test_lens_fails_per_missing_surface(
    tmp_path: Path, drop: str, path_fragment: str, token_fragment: str
) -> None:
    _write_lens_corpus(tmp_path, drop=drop)
    errors = check_codex_concerns_persistence_lens(repo_root=tmp_path)
    assert errors, f"drop={drop!r} must FAIL"
    joined = "\n".join(errors)
    assert path_fragment in joined
    assert token_fragment in joined


def test_lens_fails_on_missing_ledger_line(tmp_path: Path) -> None:
    _write_lens_corpus(tmp_path, drop="reviewer-ledger-line")
    errors = check_codex_concerns_persistence_lens(repo_root=tmp_path)
    assert len(errors) == 1
    assert "Prior-concerns ledger:" in errors[0]


def test_lens_passes_on_live_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binds the landed #2326 edits; the standing regression guard for
    future refactors of any of the four surfaces."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    errors = check_codex_concerns_persistence_lens(repo_root=None)
    assert errors == [], f"live tree should carry all four surfaces; got: {errors}"


def test_check_codex_concerns_persistence_bundled_in_no_flags(tmp_path: Path) -> None:
    """Two-part behavioral bundling pin (the #1701 test's precedent shape;
    source-pin part per tests/test_workflow_lint_smoke_blind_spots.py).

    Part A — scoped-flag subprocess against a DRIFTED corpus (SKILL.md
    subsection dropped), rooted via ``EPS_WORKFLOW_LINT_REPO_ROOT``: proves
    the flag exists, the dispatch calls the function, and it emits its
    uniquely-tagged error (nonzero exit).

    Part B — no-flags OR-chain + dispatch-ladder evidence: ``main()``'s
    source names ``args.check_codex_concerns_persistence`` in BOTH the
    ``no_flags = not (...)`` OR-chain and the ``or no_flags`` dispatch
    ladder — the #1385 / #1648 silent-unbundling shape stays pinned across
    a later dispatch refactor.
    """
    # Part A — scoped-flag subprocess against a drifted corpus.
    _write_lens_corpus(tmp_path, drop="skill-heading")
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
            "--check-codex-concerns-persistence",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "Codex concerns persistence at verdict collection" in combined, (
        "the #2326 error token is missing from the output — the CLI flag "
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
    assert "args.check_codex_concerns_persistence" in or_chain_src, (
        "args.check_codex_concerns_persistence is NOT in the no_flags "
        "OR-chain — a bare workflow_lint.py invocation will not fire this "
        f"check. OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_codex_concerns_persistence or no_flags" in main_src, (
        "args.check_codex_concerns_persistence is NOT dispatched under "
        "`or no_flags` — the flag is defined but not bundled into the "
        "no-flags default run."
    )
