"""Tests for the #1491/#2144 null-statistic gate-calibration surface pin in
``scripts/workflow_lint.py``.

One check under test: ``check_null_gate_calibration_lens`` (the FAIL surface
pin, ``--check-null-gate-calibration-lens``, bundled into the no-flags
default run). The lens must stay present across its SIX surfaces:

(1) the ``## Gate thresholds on a NULL statistic need a MEASURED calibration
    basis`` H2 in selection-symmetric-nulls.md;
(2) the ``Measured calibration basis for NULL-statistic gates`` bullet in
    planner-section-reference.md § 7 (region-anchored);
(3) the ``MEASURED 1-cell calibration pilot`` capsule token in planner.md §7
    (region-anchored);
(4) ``null-statistic gate`` AND ``defaults to ADVISORY`` in the
    critic-lens-reference.md Statistics & Measurement lens region;
(5) ``null-statistic gate calibration`` in critic.md;
(6) ``null-statistic gate`` in the statistics-critic.md item-11 region.

Fixture classes (plan #2144 §4.10):

- green corpus (all six surfaces present) → ``[]``;
- one red fixture per pinned surface (token stripped in a tmp corpus ⇒ a
  FAIL naming that file);
- REGION-ESCAPE fixtures: the token present in the file but OUTSIDE the
  anchored region ⇒ still FAILs (a bare whole-file substring search would
  let the token drift into an unrelated section and pass);
- the IN-REGION-COLLISION fixture (critic F2): the new clause's
  advisory-default sentence REMOVED while a pre-existing in-region
  lowercase ``advisory`` line (the critic-lens-reference.md line-822 shape,
  "its gates are advisory monitoring thresholds rather than pass/fail")
  remains ⇒ the check must still FAIL — this is the fixture that would
  have caught plan v2's ``advisory`` token choice, which could never fail
  independently;
- the live-tree green pin (binds the landed #2144 edits);
- the no-flags bundling pin (the check runs in the default dispatch set).

Incident #1491: Gate 1 was pre-registered as ``abs(r2_null) < 0.05`` on a
shuffle-refit null whose realized values ran -1 to -4 (unsatisfiable by
construction — a refit null is strictly negative, -d/(n-d-1)-scale, depth
non-monotone in the shape parameters); all 8 shards of the 0.5B rung
hard-aborted on an 8xH200 pod, and the asserted ``null_floor = -3.0`` died
the same way at the 1.5B rung (realized -3.40 to -3.80).
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

from workflow_lint import check_null_gate_calibration_lens  # noqa: E402

_H2 = "## Gate thresholds on a NULL statistic need a MEASURED calibration basis"
_PSR_TOKEN = "Measured calibration basis for NULL-statistic gates"
# BODY-unique companion to _PSR_TOKEN (#2144 code-review round 1): the heading
# token occurs twice in-region (sub-section heading + the §4.3 cross-ref), so
# pinning it alone is satisfiable by the cross-ref while the whole sub-section
# body is stripped. This token lives only inside the sub-section's own prose.
_PSR_BODY_TOKEN = "re-calibrate at first null draw"
_PLANNER_TOKEN = "MEASURED 1-cell calibration pilot"
_GATE_TOKEN = "null-statistic gate"
_ADVISORY_TOKEN = "defaults to ADVISORY"
_CRITIC_TOKEN = "null-statistic gate calibration"

# The pre-existing in-region collision line (the critic-lens-reference.md
# line-822 shape) — ALWAYS present in the corpus lens region so the
# collision fixture reproduces the v2 token-choice failure mode.
_COLLISION_LINE = "its gates are advisory monitoring thresholds rather than pass/fail"


def _write_corpus(root: Path, *, drop: str | None = None) -> Path:
    """Build a minimal six-surface corpus under ``root``; ``drop`` removes
    (or relocates, for the ``*-escape`` cases) exactly one surface/token."""
    agents = root / ".claude" / "agents"
    rules = root / ".claude" / "rules"
    agents.mkdir(parents=True, exist_ok=True)
    rules.mkdir(parents=True, exist_ok=True)

    # (1) the rule file + H2.
    if drop != "rule-file":
        heading = "" if drop == "rule-heading" else _H2 + "\n\nBody of the clause.\n"
        (rules / "selection-symmetric-nulls.md").write_text(
            "# Selection-symmetric nulls\n\n"
            "## Band-vs-ceiling informativeness check\n\nExisting content.\n\n"
            + heading
            + "\n## Noise-structure symmetry (shared-baseline difference vectors)\n\nContent.\n",
            encoding="utf-8",
        )

    # (2) planner-section-reference.md: the ## 7. Decision Gates region.
    # TWO pinned tokens: the sub-section HEADING and a BODY-unique phrase.
    # The heading alone is insufficient (it also appears in the §4.3 cross-ref),
    # so the drop cases exercise each half independently.
    psr_heading = f"**{_PSR_TOKEN} (#1491).** Full bullet text.\n"
    psr_body = f"Not yet materialized ⇒ mark the band `inferred — {_PSR_BODY_TOKEN}`.\n"
    psr_bullet = psr_heading + psr_body
    if drop in ("psr-token", "psr-escape"):
        in_region = ""
    elif drop == "psr-heading-only":
        in_region = psr_body
    elif drop == "psr-body-only":
        in_region = psr_heading
    else:
        in_region = psr_bullet
    out_of_region = psr_bullet if drop == "psr-escape" else ""
    (rules / "planner-section-reference.md").write_text(
        "## 7. Decision Gates\n\nGate grounding content.\n\n"
        + in_region
        + "\n## 9. Resources & Parallelism\n\nSizing content.\n"
        + out_of_region,
        encoding="utf-8",
    )

    # (3) planner.md: the ### 7. Decision Gates capsule region.
    planner_capsule = f"A null-derived gate cites a {_PLANNER_TOKEN} of that null.\n"
    in_region = "" if drop in ("planner-capsule", "planner-escape") else planner_capsule
    out_of_region = planner_capsule if drop == "planner-escape" else ""
    (agents / "planner.md").write_text(
        "# planner\n\n### 7. Decision Gates\n\nDefault to NO gates.\n"
        + in_region
        + "\n### 8. Kill criteria\n\nOther content.\n"
        + out_of_region,
        encoding="utf-8",
    )

    # (4) critic-lens-reference.md: the Statistics & Measurement lens region,
    # ALWAYS carrying the pre-existing lowercase-advisory collision line.
    gate_part = (
        ""
        if drop in ("clr-gate-token", "clr-escape")
        else (f"ALSO verify {_GATE_TOKEN} calibration: an asserted constant is a REVISE.\n")
    )
    advisory_part = (
        ""
        if drop in ("clr-advisory-token", "clr-escape")
        else (f"A null-side condition {_ADVISORY_TOKEN} logging; hard-abort is opt-in.\n")
    )
    escaped = (
        f"ALSO verify {_GATE_TOKEN} calibration.\nA null-side condition "
        f"{_ADVISORY_TOKEN} logging.\n"
        if drop == "clr-escape"
        else ""
    )
    (rules / "critic-lens-reference.md").write_text(
        "### Methodology lens\n\nItems.\n" + escaped + "\n### Statistics & Measurement lens\n\n"
        f"11. Selection-symmetric nulls. When {_COLLISION_LINE}, the read is "
        "monitoring-only.\n"
        + gate_part
        + advisory_part
        + "\n### Alternative Explanations lens\n\nItems.\n",
        encoding="utf-8",
    )

    # (5) critic.md: the item-11 capsule token.
    critic_capsule = (
        "11 selection-symmetric nulls"
        if drop == "critic-capsule"
        else f"11 selection-symmetric nulls (band vs DV ceiling; {_CRITIC_TOKEN})"
    )
    (agents / "critic.md").write_text(
        "# critic\n\n" + critic_capsule + " · 12 re-cost.\n", encoding="utf-8"
    )

    # (6) statistics-critic.md: the item-11 region.
    stats_item = (
        ""
        if drop in ("stats-item11", "stats-escape")
        else (f"    AND a registered {_GATE_TOKEN} cites a MEASURED pilot; #1491.\n")
    )
    escaped_stats = f"    Sibling note: {_GATE_TOKEN}.\n" if drop == "stats-escape" else ""
    (agents / "statistics-critic.md").write_text(
        "# statistics-critic\n\n"
        "10. Dual-DV.\n"
        "11. Selection-symmetric nulls (headline vs null band).\n"
        + stats_item
        + "12. Re-cost on power-raising recommendations.\n"
        + escaped_stats,
        encoding="utf-8",
    )
    return root


def test_passes_on_complete_corpus(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    errors = check_null_gate_calibration_lens(repo_root=tmp_path)
    assert errors == [], f"complete corpus should pass; got: {errors}"


# (drop-case, substring the error must carry, path fragment, EXACT error count).
# The exact count pins plan §6 criterion 8's "exactly one FAIL per stripped
# surface" form (#2144 code-review round 1, Minor 2) — a looser >=1 assertion
# would not notice a check that starts emitting spurious extra errors. The
# psr-token case expects 2 because dropping the whole bullet strips BOTH of
# that surface's pinned tokens.
_DROP_CASES: list[tuple[str, str, str, int]] = [
    ("rule-file", "missing", "rules/selection-symmetric-nulls.md", 1),
    (
        "rule-heading",
        "Gate thresholds on a NULL statistic",
        "rules/selection-symmetric-nulls.md",
        1,
    ),
    ("psr-token", "## 7. Decision Gates", "rules/planner-section-reference.md", 2),
    ("psr-heading-only", _PSR_TOKEN, "rules/planner-section-reference.md", 1),
    ("psr-body-only", _PSR_BODY_TOKEN, "rules/planner-section-reference.md", 1),
    ("planner-capsule", "### 7. Decision Gates", "agents/planner.md", 1),
    ("clr-gate-token", _GATE_TOKEN, "rules/critic-lens-reference.md", 1),
    ("clr-advisory-token", _ADVISORY_TOKEN, "rules/critic-lens-reference.md", 1),
    ("critic-capsule", _CRITIC_TOKEN, "agents/critic.md", 1),
    ("stats-item11", "item-11 region", "agents/statistics-critic.md", 1),
]


@pytest.mark.parametrize(("drop", "token", "path_frag", "n_expected"), _DROP_CASES)
def test_fails_per_missing_surface(
    tmp_path: Path, drop: str, token: str, path_frag: str, n_expected: int
) -> None:
    """Each pinned surface FAILs individually when its token is stripped."""
    _write_corpus(tmp_path, drop=drop)
    errors = check_null_gate_calibration_lens(repo_root=tmp_path)
    assert len(errors) == n_expected, (
        f"drop={drop}: expected exactly {n_expected} error(s); got {len(errors)}: {errors}"
    )
    assert any(token in e and path_frag in e for e in errors), (
        f"drop={drop}: no error carries both {token!r} and {path_frag!r}; got: {errors}"
    )


_ESCAPE_CASES: list[tuple[str, str]] = [
    ("psr-escape", "rules/planner-section-reference.md"),
    ("planner-escape", "agents/planner.md"),
    ("clr-escape", "rules/critic-lens-reference.md"),
    ("stats-escape", "agents/statistics-critic.md"),
]


@pytest.mark.parametrize(("drop", "path_frag"), _ESCAPE_CASES)
def test_region_escape_still_fails(tmp_path: Path, drop: str, path_frag: str) -> None:
    """A token present in the file but OUTSIDE the anchored region still
    FAILs — region-anchoring is the point (a whole-file substring search
    would pass on drifted tokens)."""
    _write_corpus(tmp_path, drop=drop)
    errors = check_null_gate_calibration_lens(repo_root=tmp_path)
    assert errors, f"drop={drop}: expected >=1 error (token is outside the region)"
    assert any(path_frag in e for e in errors), (
        f"drop={drop}: no error names {path_frag!r}; got: {errors}"
    )


def test_in_region_collision_still_fails(tmp_path: Path) -> None:
    """The critic-F2 fixture: the corpus lens region carries the
    pre-existing lowercase ``advisory`` line (the line-822 shape) while the
    new clause's advisory-default sentence is REMOVED — the check must
    still FAIL. Plan v2 pinned bare ``advisory``, which this collision
    satisfies vacuously; the landed ``defaults to ADVISORY`` pin does not."""
    _write_corpus(tmp_path, drop="clr-advisory-token")
    clr = tmp_path / ".claude" / "rules" / "critic-lens-reference.md"
    text = clr.read_text(encoding="utf-8")
    lens_region = text[text.find("### Statistics & Measurement lens") :]
    lens_region = lens_region[: lens_region.find("\n### ")]
    assert "advisory" in lens_region, "fixture must keep the lowercase collision line in-region"
    assert _ADVISORY_TOKEN not in lens_region
    errors = check_null_gate_calibration_lens(repo_root=tmp_path)
    assert any(_ADVISORY_TOKEN in e and "critic-lens-reference.md" in e for e in errors), (
        f"the in-region lowercase 'advisory' collision must not satisfy the "
        f"{_ADVISORY_TOKEN!r} pin; got: {errors}"
    )


def test_passes_on_live_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binds the landed #2144 edits; the standing regression guard for
    future refactors of any of the six surfaces."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    errors = check_null_gate_calibration_lens(repo_root=None)
    assert errors == [], f"live tree should carry all six surfaces; got: {errors}"


def test_check_null_gate_calibration_lens_bundled_in_no_flags(tmp_path: Path) -> None:
    """Two-part behavioral bundling pin (the #2165 sibling test's shape).

    Part A — scoped-flag subprocess against a DRIFTED corpus (rule file
    dropped), rooted via ``EPS_WORKFLOW_LINT_REPO_ROOT``: proves the flag
    exists, the dispatch calls the function, and it emits its error
    (nonzero exit).

    Part B — no-flags OR-chain + dispatch-ladder evidence: ``main()``'s
    source names ``args.check_null_gate_calibration_lens`` in BOTH the
    ``no_flags = not (...)`` OR-chain and the ``or no_flags`` dispatch
    ladder.
    """
    # Part A — scoped-flag subprocess against a drifted corpus.
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
            "--check-null-gate-calibration-lens",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "selection-symmetric-nulls.md" in combined, (
        "null-gate error token missing from output — the CLI flag does not "
        f"dispatch the check. exit={result.returncode}, combined output:\n{combined}"
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
    assert "args.check_null_gate_calibration_lens" in or_chain_src, (
        "args.check_null_gate_calibration_lens is NOT in the no_flags "
        "OR-chain — a bare workflow_lint.py invocation will not fire this "
        f"check. OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_null_gate_calibration_lens or no_flags" in main_src, (
        "args.check_null_gate_calibration_lens is NOT dispatched under "
        "`or no_flags` — the flag is defined but not bundled into the "
        "no-flags default run."
    )
