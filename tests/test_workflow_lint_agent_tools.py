"""Mutation tests for ``workflow_lint.py --check-agent-tools`` (task #840).

The check enforces the explicit-tool-surface invariant on
``.claude/agents/*.md``: (1) every file declares ``tools:`` or
``disallowedTools:``; (2) every spec-BODY tool mention per the widened
extractor (``mcp__...`` tokens, built-in literals, ``Agent``/``Skill``
phrase forms, prose MCP aliases) is covered by the declaration, modulo
``AGENT_TOOLS_MENTION_EXCEPTIONS``; (2b) every DECLARED ``mcp__...`` token
names a server in ``KNOWN_MCP_SERVERS`` (silent-typo guard); (3) a
denylist never denies a body-mentioned tool.

Motivating incident: #778 — an agent file with no ``tools:`` key inherits
the parent session's full MCP tool-schema payload (~168K static first-turn
tokens measured worst case) and dies in autocompact thrash.

Each mutation test pins one FAIL branch so a future refactor cannot
silently gut the check while the current tree stays green.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LINT = _REPO_ROOT / "scripts" / "workflow_lint.py"
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint  # noqa: E402
from workflow_lint import check_agent_tools  # noqa: E402


def _agent_file(tmp_path: Path, name: str, frontmatter: str, body: str) -> Path:
    """Write a minimal synthetic agent file and return its path."""
    path = tmp_path / name
    path.write_text(f"---\nname: {name.removesuffix('.md')}\n{frontmatter}\n---\n\n{body}\n")
    return path


# ---------------------------------------------------------------------------
# Current tree PASSes (the shipped declarations + exceptions dict are
# calibrated so the real agent files lint clean).
# ---------------------------------------------------------------------------


def test_current_tree_passes() -> None:
    errors = check_agent_tools()
    assert errors == [], "\n".join(errors)


def test_cli_flag_registered_and_passes_on_current_tree() -> None:
    proc = subprocess.run(
        [sys.executable, str(_LINT), "--check-agent-tools"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stderr


# ---------------------------------------------------------------------------
# Check 1 — declaration required.
# ---------------------------------------------------------------------------


def test_missing_declaration_fails(tmp_path: Path) -> None:
    _agent_file(tmp_path, "fake.md", "effort: xhigh", "Read files and report.")
    errors = check_agent_tools(roots=[tmp_path])
    assert len(errors) == 1
    assert "neither 'tools:' nor" in errors[0]
    assert "fake.md" in errors[0]


def test_declared_file_passes(tmp_path: Path) -> None:
    _agent_file(
        tmp_path,
        "fake.md",
        "tools:\n  - Read\n  - Bash",
        "Read files and report.",
    )
    assert check_agent_tools(roots=[tmp_path]) == []


# ---------------------------------------------------------------------------
# Check 2 — mentioned tools must be covered by the allowlist.
# ---------------------------------------------------------------------------


def test_unlisted_mcp_token_fails(tmp_path: Path) -> None:
    _agent_file(
        tmp_path,
        "fake.md",
        "tools:\n  - Read",
        "Call `mcp__foo__bar` to fetch the data.",
    )
    errors = check_agent_tools(roots=[tmp_path])
    assert any("mcp__foo__bar" in e and "does not cover it" in e for e in errors), errors


def test_agent_phrase_without_agent_listed_fails(tmp_path: Path) -> None:
    _agent_file(
        tmp_path,
        "fake.md",
        "tools:\n  - Read\n  - Bash",
        "Spawn `code-reviewer` via the `Agent` tool when the diff is ready.",
    )
    errors = check_agent_tools(roots=[tmp_path])
    assert any("'Agent'" in e for e in errors), errors


def test_context7_alias_without_server_listed_fails(tmp_path: Path) -> None:
    _agent_file(
        tmp_path,
        "fake.md",
        "tools:\n  - Read\n  - Bash",
        "Verify the API by reading docs via the `context7` MCP server.",
    )
    errors = check_agent_tools(roots=[tmp_path])
    assert any("mcp__plugin_context7_context7" in e for e in errors), errors


def test_server_level_declaration_covers_full_tool_mention(tmp_path: Path) -> None:
    _agent_file(
        tmp_path,
        "fake.md",
        "tools:\n  - Read\n  - mcp__ssh",
        "Launch with `mcp__ssh__ssh_execute` and tail via SSH MCP.",
    )
    assert check_agent_tools(roots=[tmp_path]) == []


# ---------------------------------------------------------------------------
# Check 2b — declared-name validity (silent-typo guard).
# ---------------------------------------------------------------------------


def test_typoed_declared_server_fails(tmp_path: Path) -> None:
    _agent_file(
        tmp_path,
        "fake.md",
        "tools:\n  - Read\n  - mcp__plugin_context7_contex7",
        "Read files and report.",
    )
    errors = check_agent_tools(roots=[tmp_path])
    assert any(
        "mcp__plugin_context7_contex7" in e and "not a known MCP server" in e for e in errors
    ), errors


def test_known_server_declaration_passes_validity(tmp_path: Path) -> None:
    _agent_file(
        tmp_path,
        "fake.md",
        "tools:\n  - Read\n  - mcp__plugin_context7_context7",
        "Read files and report.",
    )
    assert check_agent_tools(roots=[tmp_path]) == []


# ---------------------------------------------------------------------------
# Check 3 — a denylist must not deny a body-mentioned tool.
# ---------------------------------------------------------------------------


def test_denylist_denying_mentioned_tool_fails(tmp_path: Path) -> None:
    _agent_file(
        tmp_path,
        "fake.md",
        "disallowedTools: mcp__ssh, mcp__todoist",
        "Run `mcp__ssh__ssh_execute` on the pod, then summarize.",
    )
    errors = check_agent_tools(roots=[tmp_path])
    assert any("mcp__ssh__ssh_execute" in e and "denylist" in e for e in errors), errors


def test_denylist_file_inherits_everything_not_denied(tmp_path: Path) -> None:
    # A denylist-only file skips the allowlist-containment check: mentioning
    # WebSearch (not denied) is fine even though no `tools:` key exists.
    _agent_file(
        tmp_path,
        "fake.md",
        "disallowedTools: mcp__todoist, mcp__google-workspace",
        "Use WebSearch to sanity-check the claim, then report.",
    )
    assert check_agent_tools(roots=[tmp_path]) == []


# ---------------------------------------------------------------------------
# Exceptions dict — a waived (file, token) pair PASSes.
# ---------------------------------------------------------------------------


def test_exception_dict_waives_mention(tmp_path: Path, monkeypatch) -> None:
    _agent_file(
        tmp_path,
        "fake.md",
        "tools:\n  - Read",
        "The upload-verifier runs `mcp__ssh__ssh_execute` on the pod (not us).",
    )
    # Without the exception: FAIL.
    assert check_agent_tools(roots=[tmp_path]) != []
    # With the (file, token) waiver: PASS.
    monkeypatch.setattr(
        workflow_lint,
        "AGENT_TOOLS_MENTION_EXCEPTIONS",
        {("fake.md", "mcp__ssh__ssh_execute"): "describes another actor's tool use"},
    )
    assert check_agent_tools(roots=[tmp_path]) == []


def test_shipped_exceptions_all_carry_reasons() -> None:
    for (fname, token), reason in workflow_lint.AGENT_TOOLS_MENTION_EXCEPTIONS.items():
        assert fname.endswith(".md"), (fname, token)
        assert isinstance(reason, str) and len(reason) >= 10, (
            f"exception ({fname}, {token}) must carry a real inline reason"
        )
