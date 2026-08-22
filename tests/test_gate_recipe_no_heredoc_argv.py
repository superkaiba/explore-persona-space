"""Durability pin for the #2115 prong-2 gate-launcher composition rule.

The Step 10d pre-push lint-gate and surgical-twin launcher recipes in
`.claude/skills/issue/SKILL.md` used to compose their gate script via a
`cat > "$..._SCRIPT" <<'...EOF'` heredoc INSIDE the launcher Bash call.
That ships the entire multi-KB gate workload as Bash tool-call argv
through the harness transport — the surface where ~12-18 autonomous
sessions' gate dispatches lost their tool_result (task #2115) — and the
PreToolUse guards scan the full argv including heredoc bodies (#1756).
The recipes now compose the script file with the WRITE TOOL as its own
prior step; the launcher bg-Bash carries only the tiny chmod + setsid
launch. This test pins both properties per gate region so a future edit
cannot silently reintroduce the heredoc-argv shape:

1. no gate-script heredoc composition line (`cat > "$VAR" <<...` /
   `cat > /tmp/... <<...`) anywhere in the region — the prose BAN
   mentions (`` `cat > ... <<'EOF'` ``, literal dots) deliberately do
   not match;
2. the region names the Write tool as the composition step.

Update the pinned substrings in the SAME DIFF as any SKILL.md
restructure that renames the section headings below.
"""

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

# (region name, start anchor, end anchor) — start inclusive, end exclusive.
_REGIONS = (
    (
        "step9c-test-verdict-gate",
        "**9c. Test-verdict gate (code-change paths only, inline)**",
        "#### Step 10 step 0: Completion audit",
    ),
    (
        "step10d-lint-gate",
        "#### Pre-push workflow-lint gate",
        "#### The auto-merge procedure",
    ),
    (
        "step10d-surgical-twin",
        "#### The artifact-confirmed merge procedure",
        "#### Post-merge stale-task-folder guard",
    ),
)

# A REAL gate-script heredoc composition targets a shell variable or a /tmp
# path: `cat > "$LINT_GATE_SCRIPT" <<'LINT_GATE_EOF'`. The ban-prose
# mentions read `cat > ... <<'EOF'` (literal dots) and must NOT match.
# `cat`/`tee` writing a gate script from a heredoc. The var form must accept
# BOTH `$LINT_GATE_SCRIPT` and the brace spelling `${LINT_GATE_SCRIPT}` —
# `\$[A-Za-z_]` alone misses `${`, so a brace-form revert would evade this
# assertion entirely (code-review #2115 finding 1). `tee` is covered because it
# is the other common write-from-heredoc verb.
# `tee` takes its target as an ARGUMENT, not via a redirect, so the `>` must be
# part of the cat alternative only — a shared `>>?` silently fails to match any
# tee form.
_HEREDOC_COMPOSE = re.compile(r"""(?:cat\s*>>?|tee)\s*("?\$\{?[A-Za-z_]|/tmp/)\S*\s*<<""")


def _region(name: str, start: str, end: str) -> str:
    text = issue_skill_text()
    i = text.find(start)
    assert i != -1, f"{name}: start anchor not found — update the pinned anchor: {start!r}"
    j = text.find(end, i)
    assert j != -1, f"{name}: end anchor not found — update the pinned anchor: {end!r}"
    return text[i:j]


def test_no_gate_script_heredoc_in_any_gate_region():
    for name, start, end in _REGIONS:
        region = _region(name, start, end)
        hits = [ln for ln in region.splitlines() if _HEREDOC_COMPOSE.search(ln)]
        assert not hits, (
            f"{name}: gate-script heredoc composition reintroduced (#2115 — the "
            f"heredoc body rides the launcher Bash argv through the harness "
            f"transport; compose the script with the Write tool instead): {hits}"
        )


def test_pin_regex_catches_every_revert_spelling():
    """The pin must fire on the spellings a revert could plausibly use.

    Guards the pin itself: a regex that misses `${VAR}` or `tee` lets a
    brace-form or tee-form revert land silently (code-review #2115 finding 1).
    """
    must_match = [
        """  cat > "$LINT_GATE_SCRIPT" <<'LINT_GATE_EOF'""",
        """  cat > "${LINT_GATE_SCRIPT}" <<'LINT_GATE_EOF'""",
        """  cat > ${SURGICAL_SCRIPT} <<'SURGICAL_EOF'""",
        """  tee "$LINT_GATE_SCRIPT" <<'LINT_GATE_EOF'""",
        """  cat >> /tmp/issue-2115-lint-gate.sh <<'EOF'""",
    ]
    for line in must_match:
        assert _HEREDOC_COMPOSE.search(line), (
            f"pin regex fails to match a revert spelling — a revert using this "
            f"form would evade the ban assertion: {line!r}"
        )
    # The ban-PROSE mentions use literal dots and must stay unmatched, or the
    # pin fires on the very documentation telling people not to do this.
    must_not_match = [
        """  # never `cat > ... <<'EOF'` — the body rides the argv""",
        """  # compose the script with the Write tool, not cat > ... << EOF""",
    ]
    for line in must_not_match:
        assert not _HEREDOC_COMPOSE.search(line), (
            f"pin regex matches ban-prose — it would fire on the documentation "
            f"rather than on a real revert: {line!r}"
        )


def test_each_gate_region_names_the_write_tool():
    for name, start, end in _REGIONS:
        region = _region(name, start, end)
        assert "Write tool" in region, (
            f"{name}: the region no longer names the Write tool as the gate-script "
            f"composition step (#2115 prong 2) — restore the composition rule."
        )


def test_launcher_regions_show_write_composition_shape():
    """The two script-file launchers show the concrete Write(file_path=...)
    STEP 1 / launcher-only STEP 2 shape (the copy-paste surface)."""
    for name, start, end in _REGIONS[1:]:
        region = _region(name, start, end)
        assert "Write(file_path=$" in region, f"{name}: Write(file_path=...) shape missing"
        assert "launcher-only bg-Bash" in region, f"{name}: STEP 2 launcher-only marker missing"
