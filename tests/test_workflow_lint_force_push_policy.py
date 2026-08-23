"""Tests for ``workflow_lint.check_force_push_policy_lens`` (#2313).

The FAIL surface pin (``--check-force-push-policy``, bundled into the
no-flags default run): the #2313 force-push ruling must stay present at its
definition site — ``.claude/rules/auto-continuation.md``,
STATE-TO-``blocked`` criterion 2 (region-anchored on the criterion-2 bullet,
up to criterion 3). Four required tokens:

1. the literal ``--force-with-lease`` (acceptance A1: grep-findable under
   ``.claude/rules/`` — previously ZERO hits);
2. the ``NO autonomous carve-out`` token (the ban admits no autonomous
   exception; relaxing it is a USER grant made by amending the criterion);
3. the pointer to the force-free ``Rewritten-branch landing route``
   (18-step-10d.md, #2312) — a ban without its sanctioned alternative
   strands a gate-PASSed task with no landing;
4. the precedent-reconciliation issue ids on BOTH sides of the divergent
   record (#2171/#1999 vs #2181/#2318). The ids are pinned, NEVER the
   mention COUNT — the count grows whenever another task's marker names
   the flag (plan v3 SF1).

Five stripped-corpus fixtures (4 per-token strips + the missing-file case),
each proving the check binds per-token rather than passing vacuously; plus
the live-tree PASS bind and the two-part no-flags bundling pin (the
tests/test_workflow_lint_codex_concerns_persistence.py precedent shape).
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

from workflow_lint import check_force_push_policy_lens  # noqa: E402


def _rule_text(
    *,
    lease_token: bool = True,
    carveout_token: bool = True,
    route_pointer: bool = True,
    precedent_ids: bool = True,
    ruling_after_region: bool = False,
) -> str:
    """Compose a minimal auto-continuation.md corpus; each kwarg strips ONE
    pinned token. ``ruling_after_region`` relocates the whole ruling BELOW
    criterion 3 (region-anchoring negative control: every token present in
    the FILE, none in the criterion-2 REGION)."""
    lease = "`--force-with-lease`" if lease_token else "the lease form"
    carveout = "there is NO autonomous carve-out." if carveout_token else "no exception applies."
    route = (
        '`.claude/skills/issue/steps/18-step-10d.md` § "Rewritten-branch '
        'landing route (#2312 — force-free)"'
        if route_pointer
        else "the documented force-free procedure"
    )
    precedent = (
        "Precedent reconciled: #2171 and #1999 recorded the lease form as "
        "correct; #2181 and #2318 recorded it as a violation — the "
        "correct reading.\n"
        if precedent_ids
        else ""
    )
    ruling = (
        "     **Force-push ruling (#2313 — resolved toward the ban).** "
        f"No `/issue` path force-pushes; this holds for {lease} on the "
        "session's OWN rebased issue branch, and " + carveout + " The "
        f"sanctioned landing is the force-free route: {route}. " + precedent
    )
    parts = [
        "# Auto-continuation policy — gates, halt criteria, escalation\n\n",
        "**STATE-TO-`blocked` criteria.** Block ONLY when:\n",
        "  1. **Factual question only the user knows** — prose.\n",
        "  2. **Outside-the-worktree state mutation** — security "
        "boundary, irreversible writes (deletion, force-push, credential "
        "changes — always ask).\n",
    ]
    if not ruling_after_region:
        parts.append(ruling + "\n")
    parts.append("  3. **Public API contract change** — status enum.\n")
    if ruling_after_region:
        parts.append(ruling + "\n")
    parts.append("  4. **Step 10 completion-audit incomplete** — prose.\n")
    return "".join(parts)


_RULE_OK = _rule_text()

_RULE_VARIANTS: dict[str, str] = {
    "token-lease": _rule_text(lease_token=False),
    "token-carveout": _rule_text(carveout_token=False),
    "token-route": _rule_text(route_pointer=False),
    "token-precedent": _rule_text(precedent_ids=False),
    "ruling-after-region": _rule_text(ruling_after_region=True),
}


def _write_corpus(root: Path, drop: str | None = None) -> None:
    """Write the single-surface corpus; ``drop`` names one defect.

    ``drop == "rule-file"`` writes NOTHING (the missing-file fixture)."""
    if drop == "rule-file":
        return
    rule = root / ".claude" / "rules" / "auto-continuation.md"
    rule.parent.mkdir(parents=True, exist_ok=True)
    rule.write_text(_RULE_VARIANTS.get(drop or "", _RULE_OK), encoding="utf-8")


def test_lens_passes_on_complete_corpus(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    assert check_force_push_policy_lens(repo_root=tmp_path) == []


@pytest.mark.parametrize(
    ("drop", "token_fragment"),
    [
        ("rule-file", "missing"),
        ("token-lease", "--force-with-lease"),
        ("token-carveout", "NO autonomous carve-out"),
        ("token-route", "Rewritten-branch landing route"),
        ("token-precedent", "#2171"),
    ],
)
def test_lens_fails_per_stripped_token(tmp_path: Path, drop: str, token_fragment: str) -> None:
    """The five stripped-corpus fixtures (plan v3 Change 3): each strip
    yields >= 1 error naming the file + the missing token — the check binds
    per-token, never vacuously."""
    _write_corpus(tmp_path, drop=drop)
    errors = check_force_push_policy_lens(repo_root=tmp_path)
    assert errors, f"drop={drop!r} must FAIL"
    joined = "\n".join(errors)
    assert "auto-continuation.md" in joined
    assert token_fragment in joined


def test_lens_is_region_anchored(tmp_path: Path) -> None:
    """Negative control: every token present in the FILE but BELOW the
    criterion-3 anchor -> all four token errors fire (a region-blind
    substring check would pass this corpus)."""
    _write_corpus(tmp_path, drop="ruling-after-region")
    errors = check_force_push_policy_lens(repo_root=tmp_path)
    assert len(errors) == 4, f"expected all 4 token errors, got: {errors}"


def test_lens_fails_on_lost_criterion_anchor(tmp_path: Path) -> None:
    """A reflowed/renamed criterion-2 bullet is a LOUD error (the plan's
    kill-criterion-2 shape routes to a conscious update, never a silent
    vacuous pass)."""
    rule = tmp_path / ".claude" / "rules" / "auto-continuation.md"
    rule.parent.mkdir(parents=True, exist_ok=True)
    rule.write_text("# Auto-continuation policy\n\nno criteria list here.\n", encoding="utf-8")
    errors = check_force_push_policy_lens(repo_root=tmp_path)
    assert len(errors) == 1
    assert "criterion-2 anchor" in errors[0]


def test_lens_passes_on_live_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binds the landed #2313 Change-1 edit; the standing regression guard
    for future refactors of auto-continuation.md."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    errors = check_force_push_policy_lens(repo_root=None)
    assert errors == [], f"live tree should carry the #2313 ruling; got: {errors}"


def test_check_force_push_policy_bundled_in_no_flags(tmp_path: Path) -> None:
    """Two-part behavioral bundling pin (the
    tests/test_workflow_lint_codex_concerns_persistence.py precedent shape).

    Part A — scoped-flag subprocess against a DRIFTED corpus (rule file
    absent), rooted via ``EPS_WORKFLOW_LINT_REPO_ROOT``: proves the flag
    exists, the dispatch calls the function, and it emits its uniquely
    tagged error (nonzero exit).

    Part B — no-flags OR-chain + dispatch-ladder evidence: ``main()``'s
    source names ``args.check_force_push_policy`` in BOTH the
    ``no_flags = not (...)`` OR-chain and the ``or no_flags`` dispatch
    ladder — the silent-unbundling shape stays pinned across a later
    dispatch refactor.
    """
    # Part A — scoped-flag subprocess against a drifted (empty) corpus.
    _write_corpus(tmp_path, drop="rule-file")
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
            "--check-force-push-policy",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "force-push ruling" in combined, (
        "the #2313 error token is missing from the output — the CLI flag "
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
    assert "args.check_force_push_policy" in or_chain_src, (
        "args.check_force_push_policy is NOT in the no_flags OR-chain — a "
        "bare workflow_lint.py invocation will not fire this check. "
        f"OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_force_push_policy or no_flags" in main_src, (
        "args.check_force_push_policy is NOT dispatched under `or no_flags` "
        "— the flag is defined but not bundled into the no-flags default run."
    )
