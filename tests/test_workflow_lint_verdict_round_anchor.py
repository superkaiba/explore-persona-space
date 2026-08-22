"""Tests for ``workflow_lint.py --check-verdict-round-anchor`` (#2136).

``ensemble_verdicts_present``'s version-field fallback can answer the
CURRENT review round's durable-verdict check with a PRIOR round's
sentinel-less marker whose auto-bumped ``version`` equals the later round
number (#1336: a round-3 ``epm:code-review`` PASS answered a round-4 query
two days later). The fix threads a ``since_ts`` freshness anchor
(``review_round_anchor_ts``) through the /issue SKILL.md Step 5b
mechanical snippet — the ONE form every ensemble collection site
substitutes kinds into — plus a per-site opener table. This check pins
both surfaces, region-anchored on the durable-verdict-first rule, so a
future edit cannot silently revert to the unanchored call (the #606
copy-list-omission class).

1. ``test_passes_on_complete_corpus`` — a minimal SKILL.md carrying the
   anchored snippet + all four table rows passes.
2. ``test_fails_per_missing_surface`` — dropping the anchor kwarg, any of
   the four table rows, the region heading, or the file itself each emits
   a targeted error.
3. ``test_passes_on_live_tree`` — binds the landed #2136 edits; the
   standing regression guard.
4. ``test_check_verdict_round_anchor_bundled_in_no_flags`` — two-part
   behavioral bundling pin (the #1701 test's precedent shape).
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

from workflow_lint import check_verdict_round_anchor  # noqa: E402

_ROWS = (
    "| Step 5b (code review) | `epm:code-review[-codex]` | "
    '`("epm:experiment-implementation", "epm:results")` |',
    "| Step 9a (interpretation) | `epm:interp-critique[-codex]` | "
    '`("epm:interpretation", "epm:analysis")` |',
    "| Step 9a-bis (clean result) | `epm:clean-result-critique[-codex]` | "
    '`("epm:interpretation", "epm:analysis")` |',
    "| Step 9b-VC (redundancy screen) | `epm:followup-value-critique[-codex]` | "
    "omit `since_ts` — single-pass site |",
)


def _write_corpus(root: Path, *, drop: str | None = None) -> Path:
    """Build a minimal one-surface corpus under ``root``; ``drop`` removes
    exactly one surface/token to exercise each per-surface error."""
    skill_dir = root / ".claude" / "skills" / "issue"
    skill_dir.mkdir(parents=True, exist_ok=True)
    heading = "**Durable-verdict-first rule (fires at EVERY ensemble verdict collection).**\n\n"
    if drop == "region-heading":
        heading = "**Some other rule.**\n\n"
    snippet_call = (
        "since_ts=review_round_anchor_ts(\n"
        '       ev, opening_kinds=("epm:experiment-implementation", "epm:results"))'
    )
    if drop == "anchor-kwarg":
        snippet_call = ""
    rows = [row for i, row in enumerate(_ROWS) if drop != f"table-row-{i}"]
    table = (
        "   | collection site | kinds queried | `opening_kinds` to pass |\n"
        "   |---|---|---|\n" + "\n".join(f"   {row}" for row in rows) + "\n"
    )
    if drop != "skill-file":
        (skill_dir / "SKILL.md").write_text(
            "# /issue\n\n**5b. Read both markers.**\n\n"
            + heading
            + "1. Re-read canonical task state:\n\n"
            "   ```bash\n"
            "   uv run python - <<'PY'\n"
            "   import json\n"
            "   from explore_persona_space.task_workflow import (\n"
            "       ensemble_verdicts_present, list_events, review_round_anchor_ts)\n"
            "   ev = list_events(<N>)\n"
            "   print(json.dumps(ensemble_verdicts_present(\n"
            '       ev, ["epm:code-review", "epm:code-review-codex"], <n>,\n'
            f"       {snippet_call})))\n"
            "   PY\n"
            "   ```\n\n" + table + "\n"
            "**Autocompact-thrash respawn recipe.** Unrelated tail content.\n",
            encoding="utf-8",
        )
    return root


def test_passes_on_complete_corpus(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    errors = check_verdict_round_anchor(repo_root=tmp_path)
    assert errors == [], f"complete corpus should pass; got: {errors}"


_DROP_CASES: list[tuple[str, str]] = [
    ("skill-file", "missing"),
    ("region-heading", "Durable-verdict-first rule"),
    ("anchor-kwarg", "since_ts=review_round_anchor_ts"),
    ("table-row-0", "| Step 5b (code review) |"),
    ("table-row-1", "| Step 9a (interpretation) |"),
    ("table-row-2", "| Step 9a-bis (clean result) |"),
    ("table-row-3", "| Step 9b-VC (redundancy screen) |"),
]


@pytest.mark.parametrize(("drop", "token"), _DROP_CASES)
def test_fails_per_missing_surface(tmp_path: Path, drop: str, token: str) -> None:
    _write_corpus(tmp_path, drop=drop)
    errors = check_verdict_round_anchor(repo_root=tmp_path)
    assert errors, f"drop={drop}: expected >=1 error"
    assert any(token in e for e in errors), (
        f"drop={drop}: no error carries {token!r}; got: {errors}"
    )


def test_anchor_after_region_boundary_does_not_satisfy(tmp_path: Path) -> None:
    """A since_ts= mention BELOW the autocompact-recipe boundary must not
    satisfy the region-anchored pin (the pin keys on the snippet, not on a
    stray mention elsewhere in the file)."""
    _write_corpus(tmp_path, drop="anchor-kwarg")
    skill = tmp_path / ".claude" / "skills" / "issue" / "SKILL.md"
    skill.write_text(
        skill.read_text(encoding="utf-8")
        + "\nLater prose mentioning since_ts=review_round_anchor_ts outside the region.\n",
        encoding="utf-8",
    )
    errors = check_verdict_round_anchor(repo_root=tmp_path)
    assert any("since_ts=review_round_anchor_ts" in e for e in errors), (
        f"an out-of-region mention must not satisfy the pin; got: {errors}"
    )


def test_passes_on_live_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binds the landed #2136 edits; the standing regression guard for
    future refactors of the Step 5b durable-verdict-first surface."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    errors = check_verdict_round_anchor(repo_root=None)
    assert errors == [], f"live tree should carry the #2136 anchor surfaces; got: {errors}"


def test_check_verdict_round_anchor_bundled_in_no_flags(tmp_path: Path) -> None:
    """Two-part behavioral bundling pin (the #1701 test's precedent shape).

    Part A — scoped-flag subprocess against a DRIFTED corpus (anchor kwarg
    dropped), rooted via ``EPS_WORKFLOW_LINT_REPO_ROOT``: proves the flag
    exists, the dispatch calls the function, and it emits its
    uniquely-tagged error (nonzero exit).

    Part B — no-flags OR-chain + dispatch-ladder evidence: ``main()``'s
    source names ``args.check_verdict_round_anchor`` in BOTH the
    ``no_flags = not (...)`` OR-chain and the ``or no_flags`` dispatch
    ladder.
    """
    # Part A — scoped-flag subprocess against a drifted corpus.
    _write_corpus(tmp_path, drop="anchor-kwarg")
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
            "--check-verdict-round-anchor",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "since_ts=review_round_anchor_ts" in combined, (
        "verdict-round-anchor error token missing from output — the CLI "
        "flag does not dispatch the check. "
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
    assert "args.check_verdict_round_anchor" in or_chain_src, (
        "args.check_verdict_round_anchor is NOT in the no_flags OR-chain — "
        "a bare workflow_lint.py invocation will not fire this check. "
        f"OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_verdict_round_anchor or no_flags" in main_src, (
        "args.check_verdict_round_anchor is NOT dispatched under "
        "`or no_flags` — the flag is defined but not bundled into the "
        "no-flags default run."
    )
