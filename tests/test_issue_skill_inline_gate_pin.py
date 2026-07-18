"""Durability / contract pin for the #1500 inline-payload-lint-gate mechanism.

Pins the three-surface contract SKILL.md Step 9a-ter § Inline payload lint
gate ↔ ``.claude/hooks/guard_root_code_commit.sh`` ↔
``scripts/inline_lint_gate.py``: the prose names the mechanized gate + the
enforcing hook, the hook is registered in ``.claude/settings.json``'s Bash
matcher group, the cert-file default path literal is identical across hook and
helper (drift pin — a silent divergence would make every gate run certify into
a file the hook never reads, i.e. a permanent false-block), and the hook's
block message carries the helper invocation + the deliberate override.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL = _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
HOOK = _REPO_ROOT / ".claude" / "hooks" / "guard_root_code_commit.sh"
HELPER = _REPO_ROOT / "scripts" / "inline_lint_gate.py"
SETTINGS = _REPO_ROOT / ".claude" / "settings.json"

CERT_PATH_LITERAL = "/tmp/eps-inline-lint-cert-v1.txt"


def _gate_section() -> str:
    """The Step 9a-ter § Inline payload lint gate span of SKILL.md."""
    text = SKILL.read_text(encoding="utf-8")
    start = text.index("**Inline payload lint gate")
    # The section ends at the next numbered step heading after it.
    m = re.search(r"\n\d+\. \*\*Capture the headline", text[start:])
    assert m, "could not delimit the Inline payload lint gate section"
    return text[start : start + m.start()]


def test_step9ater_names_mechanized_gate_and_hook() -> None:
    """Durability pin (plan #1500 §6.3): the SKILL.md section names BOTH the
    mechanized gate helper and the enforcing PreToolUse hook by filename."""
    section = _gate_section()
    assert "scripts/inline_lint_gate.py" in section, "SKILL.md gate section lost the helper name"
    assert "guard_root_code_commit.sh" in section, "SKILL.md gate section lost the hook name"


def test_step9ater_keeps_prose_gate_scope_and_conservative_arm() -> None:
    """Plan §4.5 round-1 concern 3: the enforcement note must keep the
    'prose gate still binds elsewhere' scope sentence and the
    conservative-arm disclosure, so prose and mechanism never silently
    disagree."""
    section = _gate_section()
    assert "prose gate here still binds" in section, section[-500:]
    assert "blocks conservatively" in section
    assert "EPM_ALLOW_ROOT_CODE_COMMIT=1" in section


def test_settings_registers_hook_in_bash_matcher_group() -> None:
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    for entry in settings["hooks"]["PreToolUse"]:
        if entry.get("matcher") != "Bash":
            continue
        cmds = [h["command"] for h in entry.get("hooks", []) if h.get("type") == "command"]
        matches = [c for c in cmds if c.endswith("/guard_root_code_commit.sh")]
        assert len(matches) == 1, cmds
        return
    raise AssertionError("no matcher-Bash PreToolUse group in .claude/settings.json")


def test_cert_default_path_identical_in_hook_and_helper() -> None:
    """Drift pin: the hook validates the file the helper certifies into."""
    hook_text = HOOK.read_text(encoding="utf-8")
    helper_text = HELPER.read_text(encoding="utf-8")
    assert CERT_PATH_LITERAL in hook_text, "hook lost the cert default path"
    assert f'DEFAULT_CERT_PATH = "{CERT_PATH_LITERAL}"' in helper_text, (
        "helper lost the cert default path"
    )
    # Both consume the same env override, so a test/maintenance override
    # redirects producer and consumer together.
    assert "EPM_INLINE_CERT_PATH" in hook_text
    assert "EPM_INLINE_CERT_PATH" in helper_text
    # And the SKILL.md prose documents the same literal.
    assert CERT_PATH_LITERAL in _gate_section()


def test_hook_block_message_names_helper_invocation_and_override() -> None:
    hook_text = HOOK.read_text(encoding="utf-8")
    assert "scripts/inline_lint_gate.py --issue" in hook_text
    assert "EPM_ALLOW_ROOT_CODE_COMMIT=1" in hook_text
    assert "NEVER hand-write" in hook_text
