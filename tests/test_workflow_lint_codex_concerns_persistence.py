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
2.  ``test_lens_fails_per_missing_surface`` — parametrized drops, each
    naming the file + missing token.
3.  ``test_lens_passes_on_live_tree`` — binds the landed #2326 edits.
4.  ``test_check_codex_concerns_persistence_bundled_in_no_flags`` — the
    two-part behavioral bundling pin (the #1701 test's precedent shape,
    mirroring tests/test_workflow_lint_smoke_blind_spots.py).
5.  ``test_strengthened_pins_single_deterministic_error`` — round-2
    strengthening (``durability-pin-token-presence-gaps``): each
    broken-but-previously-PASSING mutation (a mid-prose-only ``CONCERN:: ``
    token, a deleted ``CONCERN:: none`` sentinel, a single collection
    invocation, a heading-only recovery clause, a dropped predicate
    sentence / resume-table preamble / 5c-ter empty-ledger literal)
    produces EXACTLY ONE deterministic lint error. Round-3 addition
    (same concern, sentinel-only alias): a composer region whose ONLY
    line-start token is a standalone ``CONCERN:: none`` — it passed the
    round-2 pins with the grammar row stripped (reconciler-executed:
    0 errors pre-fix) and now yields exactly one error per composer.
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


def _skill_text(
    *,
    forwarder_name: str = "persist_verdict_concerns.py",
    validate_invocation: bool = True,
    persist_invocation: bool = True,
    recovery_clause: bool = True,
    recovery_invocation: bool = True,
    predicate_sentence: bool = True,
    preamble: bool = True,
    empty_ledger: bool = True,
) -> str:
    """Compose a minimal SKILL.md corpus; each kwarg drops ONE pinned token."""
    inv = (
        f'    uv run python scripts/{forwarder_name} <N> --file "$MB" \\\n'
        "      --by <codex-role> --round <n> --require-block"
    )
    parts = [
        "prose above the recipe.\n\n",
        "**Codex concerns persistence at verdict collection (fires at EVERY\n"
        "marker-mode Codex verdict collection; #2326).** The Codex twins never\n"
        "mutate `concerns.jsonl`; the orchestrator forwards machine rows.\n\n",
    ]
    if validate_invocation:
        parts.append(inv + " --validate-only\n")
    if persist_invocation:
        parts.append(inv + "\n")
    parts.append("\n")
    if recovery_clause:
        parts.append("**Resume recovery (crash between marker post and persist — #2326).**\n")
        if predicate_sentence:
            parts.append(
                "Recovery binds at every resume-table row whose PREDICATE includes\n"
                "an existing current-round codex marker; run recovery FIRST.\n"
            )
        parts.append("\n")
        if recovery_invocation:
            parts.append(inv + "\n")
        parts.append("\n")
    parts.append("**5c. Apply ensemble decision rule.**\n\ndecision table here.\n\n")
    if empty_ledger:
        parts.append("If empty: log `concerns ledger: empty — nothing to walk`.\n\n")
    if preamble:
        parts.append(
            "Every row below whose PREDICATE includes an EXISTING current-round\n"
            "`epm:*-codex` marker FIRST runs the resume-recovery step (#2326).\n"
        )
    return "".join(parts)


_SKILL_OK = _skill_text()

_CODEX_CR_OK = """\
composer prose.

<!-- epm:code-review-codex v{{revision_round}} -->
**Verdict:** [PASS | FAIL]

## Concerns to persist

CONCERN:: <BLOCKER|CONCERN|NIT> <kebab-case-id> <one-line summary>

Empty set: the sole row `CONCERN:: none`.

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

# Broken-but-previously-PASSING (#2326 round 2): the CONCERN:: token appears
# only MID-PROSE (the containment clause) — the pre-strengthening substring
# check accepted this shape with the grammar row deleted.
_CODEX_CR_TOKEN_MIDLINE_ONLY = """\
composer prose.

<!-- epm:code-review-codex v{{revision_round}} -->
**Verdict:** [PASS | FAIL]

## Concerns to persist

Emit rows using the token CONCERN:: at line start; when there is nothing
to persist emit the sole row `CONCERN:: none`.

<!-- /epm:code-review-codex -->
"""

# Broken-but-previously-PASSING (#2326 round 3, `durability-pin-token-
# presence-gaps` sentinel-only alias): the grammar row is DELETED and the
# region's only line-start token is a standalone `CONCERN:: none` — it
# satisfied BOTH round-2 pins (line-start `CONCERN:: ` present; sentinel
# substring present), so the template passed with the grammar row stripped
# (the reconciler EXECUTED this corpus: 0 errors pre-fix).
_CODEX_CR_SENTINEL_ONLY = """\
composer prose.

<!-- epm:code-review-codex v{{revision_round}} -->
**Verdict:** [PASS | FAIL]

## Concerns to persist

CONCERN:: none

<!-- /epm:code-review-codex -->
"""

# Broken-but-previously-PASSING: grammar row present, `CONCERN:: none`
# empty-set sentinel deleted.
_CODEX_CR_NO_NONE = """\
composer prose.

<!-- epm:code-review-codex v{{revision_round}} -->
**Verdict:** [PASS | FAIL]

## Concerns to persist

CONCERN:: <BLOCKER|CONCERN|NIT> <kebab-case-id> <one-line summary>

<!-- /epm:code-review-codex -->
"""

_CODEX_CRC_OK = """\
composer prose.

<!-- epm:clean-result-critique-codex v{{revision_round}} -->
**Verdict:** [PASS | REVISE]

### Concerns to persist

CONCERN:: <BLOCKER|CONCERN|NIT> <kebab-case-id> <one-line summary>

Empty set: the sole row `CONCERN:: none`.

<!-- /epm:clean-result-critique-codex -->
"""

_CODEX_CRC_NO_TOKEN = """\
composer prose.

<!-- epm:clean-result-critique-codex v{{revision_round}} -->
**Verdict:** [PASS | REVISE]

<!-- /epm:clean-result-critique-codex -->
"""

_CODEX_CRC_TOKEN_MIDLINE_ONLY = """\
composer prose.

<!-- epm:clean-result-critique-codex v{{revision_round}} -->
**Verdict:** [PASS | REVISE]

### Concerns to persist

Emit rows using the token CONCERN:: at line start; when there is nothing
to persist emit the sole row `CONCERN:: none`.

<!-- /epm:clean-result-critique-codex -->
"""

_CODEX_CRC_SENTINEL_ONLY = """\
composer prose.

<!-- epm:clean-result-critique-codex v{{revision_round}} -->
**Verdict:** [PASS | REVISE]

### Concerns to persist

CONCERN:: none

<!-- /epm:clean-result-critique-codex -->
"""

_CODEX_CRC_NO_NONE = """\
composer prose.

<!-- epm:clean-result-critique-codex v{{revision_round}} -->
**Verdict:** [PASS | REVISE]

### Concerns to persist

CONCERN:: <BLOCKER|CONCERN|NIT> <kebab-case-id> <one-line summary>

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


# Per-surface drop -> corpus-text variants (each entry mutates ONE pin;
# ``None`` / any other drop leaves the surface at its OK text).
_SKILL_VARIANTS: dict[str, str] = {
    "skill-heading": "no subsection here.\n\n**5c. Apply ensemble decision rule.**\n",
    "skill-forwarder": _skill_text(forwarder_name="some_other_script.py"),
    "skill-recovery": _skill_text(recovery_clause=False),
    "skill-collection-count": _skill_text(validate_invocation=False),
    "skill-recovery-invocation": _skill_text(recovery_invocation=False),
    "skill-predicate": _skill_text(predicate_sentence=False),
    "skill-preamble": _skill_text(preamble=False),
    "skill-empty-ledger": _skill_text(empty_ledger=False),
}
_CR_VARIANTS: dict[str, str] = {
    # Tag only mid-line (prose mention), never at line start.
    "codex-cr-start-tag": ("prose Marker start tag: <!-- epm:code-review-codex v1 --> only.\n"),
    "codex-cr-token": _CODEX_CR_NO_TOKEN,
    "codex-cr-token-midline": _CODEX_CR_TOKEN_MIDLINE_ONLY,
    "codex-cr-sentinel-only": _CODEX_CR_SENTINEL_ONLY,
    "codex-cr-no-none": _CODEX_CR_NO_NONE,
}
_CRC_VARIANTS: dict[str, str] = {
    "codex-crc-start-tag": ("prose <!-- epm:clean-result-critique-codex v1 --> mid-line only.\n"),
    "codex-crc-token": _CODEX_CRC_NO_TOKEN,
    "codex-crc-token-midline": _CODEX_CRC_TOKEN_MIDLINE_ONLY,
    "codex-crc-sentinel-only": _CODEX_CRC_SENTINEL_ONLY,
    "codex-crc-no-none": _CODEX_CRC_NO_NONE,
}
_REVIEWER_VARIANTS: dict[str, str] = {
    "reviewer-step08": "### Step 0.9: Git-provenance self-check\n\nprose.\n",
    "reviewer-ledger-line": _REVIEWER_NO_LEDGER_LINE,
}


def _write_lens_corpus(root: Path, drop: str | None = None) -> None:
    """Write a minimal four-surface corpus; ``drop`` names one defect."""
    surfaces = (
        (
            root / ".claude" / "skills" / "issue" / "SKILL.md",
            "skill-file",
            _SKILL_VARIANTS,
            _SKILL_OK,
        ),
        (
            root / ".claude" / "agents" / "codex-code-reviewer.md",
            "codex-cr-file",
            _CR_VARIANTS,
            _CODEX_CR_OK,
        ),
        (
            root / ".claude" / "agents" / "codex-clean-result-critic.md",
            "codex-crc-file",
            _CRC_VARIANTS,
            _CODEX_CRC_OK,
        ),
        (
            root / ".claude" / "agents" / "code-reviewer.md",
            "reviewer-file",
            _REVIEWER_VARIANTS,
            _REVIEWER_OK,
        ),
    )
    for path, file_drop, variants, ok_text in surfaces:
        path.parent.mkdir(parents=True, exist_ok=True)
        if drop != file_drop:
            path.write_text(variants.get(drop or "", ok_text))


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


@pytest.mark.parametrize(
    ("drop", "path_fragment", "token_fragment"),
    [
        ("skill-collection-count", "SKILL.md", "COLLECTION"),
        ("skill-recovery-invocation", "SKILL.md", "carries no"),
        ("skill-predicate", "SKILL.md", "predicate-leading sentence"),
        ("skill-preamble", "SKILL.md", "resume-table preamble"),
        ("skill-empty-ledger", "SKILL.md", "empty-ledger"),
        ("codex-cr-token-midline", "codex-code-reviewer.md", "LINE-START"),
        ("codex-cr-sentinel-only", "codex-code-reviewer.md", "sentinel-only"),
        ("codex-cr-no-none", "codex-code-reviewer.md", "CONCERN:: none"),
        ("codex-crc-token-midline", "codex-clean-result-critic.md", "LINE-START"),
        ("codex-crc-sentinel-only", "codex-clean-result-critic.md", "sentinel-only"),
        ("codex-crc-no-none", "codex-clean-result-critic.md", "CONCERN:: none"),
    ],
)
def test_strengthened_pins_single_deterministic_error(
    tmp_path: Path, drop: str, path_fragment: str, token_fragment: str
) -> None:
    """Round-2 strengthening (#2326 `durability-pin-token-presence-gaps`):
    each of these mutations PASSED the pre-strengthening check (token /
    heading still present somewhere) and must now produce EXACTLY ONE
    deterministic error naming the file + the missing pin."""
    _write_lens_corpus(tmp_path, drop=drop)
    errors = check_codex_concerns_persistence_lens(repo_root=tmp_path)
    assert len(errors) == 1, f"drop={drop!r} must yield exactly one error; got: {errors}"
    assert path_fragment in errors[0]
    assert token_fragment in errors[0]


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
