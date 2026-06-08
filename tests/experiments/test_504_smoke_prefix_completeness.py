"""Task #504 round-7 grep-walker — pin v1/v2/v3 smoke-prefix-tuple completeness.

The same architectural bug class has now bitten task #504 three times in
the same task lineage:

  * Round-2 (concern_id `analyze-v2-slug-iteration`): a 1-prefix tuple
    on the analyze path silently skipped v2 smoke slugs.
  * Round-6 (epm:failure 2026-06-08T13:11:10Z): a 2-prefix tuple at
    ``cell_resolution.py:183`` + ``i504_run_cell.py:273`` silently
    skipped v3 smoke slugs at cell-build / dispatch.
  * Round-7 (epm:review-reconcile 2026-06-08T13:23:21Z): a 2-prefix
    tuple at ``scripts/i504_eval_trajectory.py:159`` silently no-opped
    the disjointness guard on v3 smoke cells — bystander ΔG could
    have been computed against a panel that included
    ``smoke_mid_band_n`` (= the persona the cell trained against).

Each round added the same fix at one new site; each round missed the
NEXT site. To stop this recurrence class going forward, this test scans
the active-path source files for the bare prefix literal
``"c504_smoke_"`` and asserts EVERY occurrence is paired with both
``"c504v2_smoke_"`` AND ``"c504v3_smoke_"`` in the same logical
statement (same line OR the next 2 lines, for tuples Black wraps across
lines), OR carries the explicit opt-out marker comment
``# epm-smoke-prefix: <reason>`` on the same line. The marker is for
v1-only paths that genuinely should NEVER widen (round-12 recovery
rigs, v1-only literal extraction, etc.); naming the reason inline is
the contract.

The test recognizes only the bare prefix ``"c504_smoke_"`` (closing
quote immediately after the trailing underscore). v1-only FULL slugs
like ``"c504_smoke_r4"`` / ``"c504_smoke_r"`` / ``"c504_smoke_lr"`` are
NOT prefix-tuple cases and are exempt from the check — they are
v1-specific data (the r-ladder picker / rank extractor / etc.) and
must not be widened.

CPU-only, sub-second; reads source files only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Files in scope: the active eval + dispatch paths for task #504 that
# could plausibly need a v1/v2/v3 prefix tuple. Frozen, NOT auto-glob —
# adding a new file is an intentional act and should be a code review.
_IN_SCOPE: tuple[Path, ...] = (
    _REPO_ROOT / "scripts" / "i504_eval_trajectory.py",
    _REPO_ROOT / "scripts" / "i504_run_cell.py",
    _REPO_ROOT / "scripts" / "i504_reval_grid.py",
    _REPO_ROOT
    / "src"
    / "explore_persona_space"
    / "experiments"
    / "contrastive_neg_geometry_504"
    / "cell_resolution.py",
)

# Regex that matches the bare prefix literal — closing quote is the
# character immediately after the trailing underscore. This is what
# distinguishes a `startswith` prefix from a full slug like `"c504_smoke_r4"`.
_BARE_PREFIX_RE = re.compile(r'"c504_smoke_"')
_V2_PREFIX_RE = re.compile(r'"c504v2_smoke_"')
_V3_PREFIX_RE = re.compile(r'"c504v3_smoke_"')

# Opt-out marker: the line must carry an inline comment of the form
# ``# epm-smoke-prefix: <reason>`` where <reason> is non-empty (the
# whole point of the marker is to NAME why the v1-only literal is
# intentional). Whitespace tolerant.
_OPTOUT_RE = re.compile(r"#\s*epm-smoke-prefix:\s*\S")


def _find_logical_statement_block(lines: list[str], idx: int) -> str:
    """Return the joined text of lines[idx:idx+3] (the bare-prefix line +
    up to 2 continuation lines).

    Black formatters wrap long tuples across multiple lines, so a 3-prefix
    tuple may legitimately span up to 3 lines. We accept any line inside
    that window providing the v2 + v3 prefixes (or the opt-out marker on
    the bare-prefix line itself).
    """
    return "\n".join(lines[idx : idx + 3])


def test_in_scope_files_exist() -> None:
    """Sanity: every in-scope file is present.

    If a file was renamed/deleted, the grep-walker would otherwise quietly
    pass with reduced coverage — this assertion makes the renaming intentional.
    """
    for path in _IN_SCOPE:
        assert path.is_file(), f"In-scope file missing: {path}"


def test_no_bare_v1_smoke_prefix_without_v2_v3_or_optout() -> None:
    """Every ``"c504_smoke_"`` literal is either part of a 3-prefix tuple
    (v1 + v2 + v3) OR carries the ``# epm-smoke-prefix: <reason>`` opt-out.

    This is the round-7 anti-recurrence pin: if a future round adds (or
    fails to widen) a 1- or 2-prefix tuple at a new site, this test
    FAILs at lint-time, NOT at pod-launch time.
    """
    violations: list[str] = []
    for path in _IN_SCOPE:
        lines = path.read_text().splitlines()
        for idx, line in enumerate(lines):
            if not _BARE_PREFIX_RE.search(line):
                continue
            # Allow the opt-out marker on the SAME line as the bare prefix.
            if _OPTOUT_RE.search(line):
                continue
            # Otherwise: the bare prefix MUST be inside a 3-prefix tuple
            # (v1 + v2 + v3) — possibly Black-wrapped across the next 2
            # lines. We accept the v2 + v3 prefixes anywhere in the
            # 3-line window starting at the bare-prefix line.
            block = _find_logical_statement_block(lines, idx)
            has_v2 = bool(_V2_PREFIX_RE.search(block))
            has_v3 = bool(_V3_PREFIX_RE.search(block))
            if not (has_v2 and has_v3):
                relpath = path.relative_to(_REPO_ROOT)
                missing = []
                if not has_v2:
                    missing.append('"c504v2_smoke_"')
                if not has_v3:
                    missing.append('"c504v3_smoke_"')
                violations.append(
                    f"{relpath}:{idx + 1}: bare prefix literal "
                    f'"c504_smoke_" found but missing {", ".join(missing)} '
                    f"in the same 3-line block, AND no "
                    f"`# epm-smoke-prefix: <reason>` opt-out marker on the "
                    f"line. Either widen the tuple to include the missing "
                    f"prefix(es), or annotate the line with the opt-out "
                    f"marker naming why this site must stay v1-only "
                    f"(e.g. round-12 recovery rig). Context line: "
                    f"{line.strip()!r}"
                )
    if violations:
        bullets = "\n  * ".join(violations)
        pytest.fail(
            'Round-7 anti-recurrence: every `"c504_smoke_"` bare prefix '
            'literal must be paired with both `"c504v2_smoke_"` and '
            '`"c504v3_smoke_"` in the same 3-line block (Black-wrapped '
            "tuples included), OR carry the inline opt-out marker "
            "`# epm-smoke-prefix: <reason>`. Violations:\n  * " + bullets
        )


def test_v2_and_v3_prefixes_only_appear_in_3prefix_tuples() -> None:
    """Every ``"c504v2_smoke_"`` and ``"c504v3_smoke_"`` literal sits next to
    the v1 literal (= part of a 3-prefix tuple, not a standalone reference).

    This is the symmetric guard: a future site that introduces JUST
    ``"c504v3_smoke_"`` (without the v1 + v2 companions) is also a bug
    class — it would mean the dispatcher's smoke path forgot the v1/v2
    cells. Catches both directions of the asymmetry.
    """
    violations: list[str] = []
    for path in _IN_SCOPE:
        lines = path.read_text().splitlines()
        for idx, line in enumerate(lines):
            for label, this_re, other_re_a, other_re_b in (
                ("c504v2_smoke_", _V2_PREFIX_RE, _BARE_PREFIX_RE, _V3_PREFIX_RE),
                ("c504v3_smoke_", _V3_PREFIX_RE, _BARE_PREFIX_RE, _V2_PREFIX_RE),
            ):
                if not this_re.search(line):
                    continue
                block = _find_logical_statement_block(lines, idx)
                if not (other_re_a.search(block) and other_re_b.search(block)):
                    relpath = path.relative_to(_REPO_ROOT)
                    violations.append(
                        f"{relpath}:{idx + 1}: prefix literal "
                        f'"{label}" found without both the v1 + the '
                        f"other vN prefix in the same 3-line block. "
                        f"Context line: {line.strip()!r}"
                    )
    if violations:
        bullets = "\n  * ".join(violations)
        pytest.fail(
            "Round-7 anti-recurrence (symmetric): every vN smoke-prefix "
            "literal must sit inside a 3-prefix tuple. Violations:\n  * " + bullets
        )


def test_known_sites_all_have_3prefix_or_optout() -> None:
    """Concrete pin: the four KNOWN sites have the expected shape.

    Hardcoded so a refactor that moves a tuple to a different line is
    caught immediately rather than silently losing coverage. If you
    intentionally moved a site, update this test together with the move.
    """
    # (relpath, expected_shape) — shape is "3prefix" or "optout"
    known_sites: tuple[tuple[str, str], ...] = (
        ("scripts/i504_eval_trajectory.py", "3prefix"),
        ("scripts/i504_run_cell.py", "3prefix"),
        ("scripts/i504_reval_grid.py", "optout"),
        (
            "src/explore_persona_space/experiments/contrastive_neg_geometry_504/cell_resolution.py",
            "3prefix",
        ),
    )
    found: dict[str, list[str]] = {}
    for path in _IN_SCOPE:
        relpath = str(path.relative_to(_REPO_ROOT))
        lines = path.read_text().splitlines()
        shapes: list[str] = []
        for idx, line in enumerate(lines):
            if not _BARE_PREFIX_RE.search(line):
                continue
            if _OPTOUT_RE.search(line):
                shapes.append("optout")
                continue
            block = _find_logical_statement_block(lines, idx)
            if _V2_PREFIX_RE.search(block) and _V3_PREFIX_RE.search(block):
                shapes.append("3prefix")
            else:
                shapes.append("incomplete")
        found[relpath] = shapes

    for relpath, expected_shape in known_sites:
        assert relpath in found, f"Known site dropped from scan: {relpath}"
        shapes = found[relpath]
        assert shapes, (
            f'Known site {relpath} no longer contains any `"c504_smoke_"` '
            "literal — if intentional, drop it from `known_sites`."
        )
        assert expected_shape in shapes, (
            f"Known site {relpath} expected shape {expected_shape!r} but "
            f"found shapes {shapes!r}. Either update the test or the source."
        )
