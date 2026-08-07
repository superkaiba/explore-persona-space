"""Tests for the #2165 smoke blind-spot checks in ``scripts/workflow_lint.py``.

Two checks under test:

- ``check_smoke_blind_spot_enumeration`` — the WARN-only best-effort AST scan
  (``--check-smoke-blind-spots``): flags smoke-conditional
  substitution/downgrade branches in named scripts when the plan carries no
  SMOKE BLIND-SPOT ENUMERATION (`.claude/rules/smoke-blind-spots.md`).
- ``check_smoke_blind_spot_review_lens`` — the FAIL surface pin
  (``--check-smoke-blind-spot-review-lens``, bundled into the no-flags
  default run): the lens must stay present across its SEVEN surfaces.

Fixtures reproduce BOTH #1336 incident shapes (plan v15 round 4 — two
consecutive production SLURM launches died on checks the pre-launch smoke
structurally bypassed), in the reshaped-inline form AND the structurally
faithful forms read from the real incident file
(``scripts/issue1336_pooled_split.py`` on the ``issue-1336-fullcorpora``
branch — NOT durable at test time, so the fixtures are self-contained):

1.  ``test_detects_substituted_implementation_shape`` — reshaped shape 1
    (SLURM 4684): early-exit toy with the production import inline after.
2.  ``test_detects_downgraded_assertion_shape`` — reshaped shape 2
    (SLURM 5005): early-return before the assert.
3.  ``test_warns_when_plan_lacks_enumeration`` — summary WARN.
4.  ``test_silent_when_plan_carries_enumeration`` — per-hit suppression.
5.  ``test_warns_when_none_escape_contradicted`` — falsified empty-form.
6.  ``test_no_warn_on_clean_script`` — shrink-only smoke param (non-trigger).
7.  ``test_unparseable_script_warns_not_crashes`` — best-effort arm.
8.  ``test_returns_empty_fail_list_always`` — WARN-only contract.
9.  ``test_review_lens_passes_on_complete_corpus`` — all seven surfaces.
10. ``test_review_lens_fails_per_missing_surface`` — 12 parametrized drops.
11. ``test_review_lens_passes_on_live_tree`` — binds the landed edits.
12. ``test_check_smoke_blind_spot_review_lens_bundled_in_no_flags`` — the
    two-part behavioral bundling pin (the #1701 test's shape).
13. ``test_detects_helper_wrapped_substitution`` — FAITHFUL shape 1: the
    production import one module-local call away (one-level resolution).
14. ``test_detects_ifelse_downgrade_shape`` — FAITHFUL shape 2: per-check
    ``if smoke: log else: raise`` with NO early exit (branch-form rule).
15. ``test_helper_resolution_is_one_level_only`` — the DISCLOSED one-level
    false-negative boundary, pinned as documented behavior.
16. ``test_detects_ternary_substitution`` — ``ast.IfExp`` branch form (plan
    section 4.10(B) names ``ast.If`` AND ``ast.IfExp``): a substituted
    implementation inside a ternary fires.
17. ``test_no_warn_on_shrink_only_ternary`` — a shrink-only ternary
    (``n = 2 if smoke else 500``) stays a non-trigger.
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

from workflow_lint import (  # noqa: E402
    check_smoke_blind_spot_enumeration,
    check_smoke_blind_spot_review_lens,
)

_TAG = "smoke-blind-spot-unenumerated"
_ESCAPE = "none — smoke executes every production gate"

# --------------------------------------------------------------------------
# Scanner fixtures (self-contained reproductions of both #1336 shapes; the
# scripts are only ast.parse'd, never executed, so the undeclared
# sentence_transformers import in fixture TEXT is inert by construction).
# --------------------------------------------------------------------------

_FIXTURE_BOTH_SHAPES = """\
import hashlib


def embed_prompts(prompts, smoke=False):
    # #1336 shape 1 (SLURM 4684), RESHAPED INLINE: smoke substitutes a toy;
    # the production import + constructor are written inline after the early
    # exit (the real code helper-wraps them -- see _FIXTURE_HELPER_WRAPPED).
    if smoke:
        return [[float(b) for b in hashlib.sha256(p.encode()).digest()] for p in prompts]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return model.encode(prompts)


def assert_split(rows, smoke=False):
    # #1336 shape 2 (SLURM 5005), RESHAPED: the gate early-returns under
    # smoke (the real code is per-check if/else -- see _FIXTURE_IFELSE_DOWNGRADE).
    if smoke:
        return
    assert len({r["split"] for r in rows}) == 3, "pooled split invariant"
"""

_FIXTURE_HELPER_WRAPPED = """\
def _load_model(revision):
    # Production import one frame down -- the REAL #1336 shape 1
    # (_load_sentence_transformer, issue-1336-fullcorpora lines 524-538).
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2", revision=revision)


def embed_prompts(prompts, revision, smoke=False):
    # Faithful #1336 shape 1: the toy branch is PURE PYTHON (no import --
    # the real code's `import numpy` inside the branch is incidental and
    # not load-bearing), and the post-exit callee is lowercase + module-local.
    if smoke:
        return [[float(len(p) % 7)] * 32 for p in prompts]
    model = _load_model(revision)
    return model.encode(prompts)
"""

_FIXTURE_IFELSE_DOWNGRADE = """\
import logging

logger = logging.getLogger(__name__)


def assert_split(manifest, smoke=False):
    # Faithful #1336 shape 2 (issue-1336-fullcorpora assert_split, lines
    # 703-790): per-check if/else log-vs-raise; NO early exit.
    corpus_locked = [cid for cid, hist in manifest.items() if len(hist) < 3]
    if corpus_locked:
        if smoke:
            logger.info("[pool] SMOKE - corpus-locked clusters (production would HALT)")
        else:
            raise AssertionError(f"corpus-locked clusters: {corpus_locked}")
"""

# Two-level variant for the one-level-boundary pin (test 15): the import
# sits TWO module-local calls away from the branch site.
_FIXTURE_TWO_LEVEL = """\
def _load_inner():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def _load_model(revision):
    # The import sits TWO calls down from the branch site.
    return _load_inner()


def embed_prompts(prompts, revision, smoke=False):
    if smoke:
        return [[float(len(p) % 7)] * 32 for p in prompts]
    model = _load_model(revision)
    return model.encode(prompts)
"""

_FIXTURE_CLEAN = """\
def run(rows, smoke=False):
    # Shrink-only smoke parameter: same code path, smaller N. The #1611 /
    # #1727 class, deliberately NOT a blind-spot trigger.
    n = 2 if smoke else 500
    return rows[:n]
"""

# Ternary (ast.IfExp) variants for the branch-form rule -- plan section
# 4.10(B) names ast.If AND ast.IfExp (round-2 fix of the
# warn-scanner-ifexp-dropped concern).
_FIXTURE_TERNARY_SUBSTITUTION = """\
def _toy(prompts):
    # Pure-python toy: no import, no capitalized constructor.
    return [[float(len(p) % 7)] * 32 for p in prompts]


def _load_model(revision):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2", revision=revision)


def embed_prompts(prompts, revision, smoke=False):
    # Ternary form of #1336 shape 1: the substitution lives in an ast.IfExp,
    # with the production import one module-local call away (one-level
    # lowercase-callee resolution must apply on the ternary arm too).
    vecs = _toy(prompts) if smoke else _load_model(revision).encode(prompts)
    return vecs
"""

_FIXTURE_TERNARY_SHRINK = """\
def run(rows, smoke=False):
    # Shrink-only ternaries: same code path on both arms, no substituted
    # implementation -- must NOT fire under the ast.IfExp branch rule.
    n = 2 if smoke else 500
    limit = min(n, len(rows)) if smoke else n
    return rows[:limit]
"""

_PLAN_WITH_ENUMERATION = """\
## Smoke run

Smoke blind-spot enumeration:
- `embed_prompts` substitutes a hash toy under smoke; the
  `sentence_transformers` import + MPNet constructor run only in production.
- `assert_split` downgrades its split gates to log lines under smoke.
"""

_PLAN_WITHOUT_ENUMERATION = """\
## Smoke run

We run a pre-launch smoke first, then the production sweep.
"""

_PLAN_WITH_ESCAPE = (
    "## Smoke run\n\nSmoke blind-spot enumeration: none — smoke executes every production gate\n"
)


def _lineno_of(src: str, needle: str, occurrence: int = 1) -> int:
    count = 0
    for i, line in enumerate(src.splitlines(), start=1):
        if needle in line:
            count += 1
            if count == occurrence:
                return i
    raise AssertionError(f"{needle!r} (occurrence {occurrence}) not in fixture")


def _scan(
    tmp_path: Path,
    fixtures: dict[str, str],
    plan_text: str | None = None,
) -> tuple[list[str], list[str], dict[str, Path]]:
    """Write fixtures under ``tmp_path``, run the scanner, return
    (result, warn_sink, name->path)."""
    paths: dict[str, Path] = {}
    for name, src in fixtures.items():
        p = tmp_path / name
        p.write_text(src, encoding="utf-8")
        paths[name] = p
    plan_path: Path | None = None
    if plan_text is not None:
        plan_path = tmp_path / "plan.md"
        plan_path.write_text(plan_text, encoding="utf-8")
    sink: list[str] = []
    result = check_smoke_blind_spot_enumeration(list(paths.values()), plan_path, warn_sink=sink)
    return result, sink, paths


# --------------------------------------------------------------------------
# Scanner: detection on both #1336 shapes (reshaped + faithful)
# --------------------------------------------------------------------------


def test_detects_substituted_implementation_shape(tmp_path: Path) -> None:
    """Reshaped shape 1 (SLURM 4684): early-exit toy, production import +
    constructor inline after the branch."""
    _, sink, paths = _scan(tmp_path, {"both.py": _FIXTURE_BOTH_SHAPES})
    lineno = _lineno_of(_FIXTURE_BOTH_SHAPES, "if smoke:", occurrence=1)
    marker = f"{paths['both.py']}:{lineno}:"
    assert any("substituted-implementation" in w and marker in w for w in sink), (
        f"expected a substituted-implementation WARN at {marker}; got: {sink}"
    )


def test_detects_downgraded_assertion_shape(tmp_path: Path) -> None:
    """Reshaped shape 2 (SLURM 5005): early-return before the assert."""
    _, sink, paths = _scan(tmp_path, {"both.py": _FIXTURE_BOTH_SHAPES})
    lineno = _lineno_of(_FIXTURE_BOTH_SHAPES, "if smoke:", occurrence=2)
    marker = f"{paths['both.py']}:{lineno}:"
    assert any("downgraded-gate" in w and marker in w for w in sink), (
        f"expected a downgraded-gate WARN at {marker}; got: {sink}"
    )


def test_detects_helper_wrapped_substitution(tmp_path: Path) -> None:
    """FAITHFUL shape 1: the toy branch is import-free, the production
    import is one module-local call away — the BASE classifier alone cannot
    fire here, so this pins the one-level lowercase-callee resolution as
    load-bearing."""
    _, sink, paths = _scan(tmp_path, {"helper.py": _FIXTURE_HELPER_WRAPPED})
    lineno = _lineno_of(_FIXTURE_HELPER_WRAPPED, "if smoke:", occurrence=1)
    marker = f"{paths['helper.py']}:{lineno}:"
    assert any("substituted-implementation" in w and marker in w for w in sink), (
        f"expected a substituted-implementation WARN at {marker} via the "
        f"one-level helper resolution; got: {sink}"
    )


def test_detects_ifelse_downgrade_shape(tmp_path: Path) -> None:
    """FAITHFUL shape 2: per-check ``if smoke: logger.info else: raise``
    with NO early exit — the BRANCH-form rule (gate on exactly the orelse
    side)."""
    _, sink, paths = _scan(tmp_path, {"ifelse.py": _FIXTURE_IFELSE_DOWNGRADE})
    lineno = _lineno_of(_FIXTURE_IFELSE_DOWNGRADE, "if smoke:", occurrence=1)
    marker = f"{paths['ifelse.py']}:{lineno}:"
    assert any("downgraded-gate" in w and marker in w for w in sink), (
        f"expected a downgraded-gate WARN at the inner if/else line {marker}; got: {sink}"
    )


def test_detects_ternary_substitution(tmp_path: Path) -> None:
    """``ast.IfExp`` branch form (plan section 4.10(B)): a smoke-conditional
    ternary substituting the production implementation fires, including
    through the one-level module-local lowercase-callee resolution on the
    production arm. Gate-downgrade has NO ternary analogue by construction
    (``assert``/``raise`` are statements), so only the
    substituted-implementation class is asserted here."""
    _, sink, paths = _scan(tmp_path, {"ternary.py": _FIXTURE_TERNARY_SUBSTITUTION})
    lineno = _lineno_of(_FIXTURE_TERNARY_SUBSTITUTION, "if smoke else")
    marker = f"{paths['ternary.py']}:{lineno}:"
    assert any("substituted-implementation" in w and marker in w for w in sink), (
        f"expected a substituted-implementation WARN at the ternary {marker}; got: {sink}"
    )


def test_no_warn_on_shrink_only_ternary(tmp_path: Path) -> None:
    """Negative arm of the ``ast.IfExp`` rule: shrink-only ternaries
    (``n = 2 if smoke else 500``) carry no implementation work on either
    arm and must stay non-triggers."""
    _, sink, _ = _scan(tmp_path, {"shrink.py": _FIXTURE_TERNARY_SHRINK})
    assert sink == [], f"shrink-only ternaries are not triggers; got: {sink}"


def test_helper_resolution_is_one_level_only(tmp_path: Path) -> None:
    """The DISCLOSED false-negative boundary: an import TWO module-local
    calls down does NOT fire (non-recursive resolution — the reviewer lens
    is the catching arm for this class; rule-file Enforcement section)."""
    _, sink, _ = _scan(tmp_path, {"twolevel.py": _FIXTURE_TWO_LEVEL})
    assert sink == [], (
        f"the one-level boundary must stay a documented false negative; got WARNs: {sink}"
    )


# --------------------------------------------------------------------------
# Scanner: plan cross-check semantics
# --------------------------------------------------------------------------


def test_warns_when_plan_lacks_enumeration(tmp_path: Path) -> None:
    _, sink, _ = _scan(tmp_path, {"both.py": _FIXTURE_BOTH_SHAPES}, _PLAN_WITHOUT_ENUMERATION)
    assert any("plan carries no smoke blind-spot enumeration" in w for w in sink), (
        f"expected the summary WARN; got: {sink}"
    )


def test_silent_when_plan_carries_enumeration(tmp_path: Path) -> None:
    """Per-hit suppression: branches are enumerated; naming-completeness is
    reviewer-owned (Step 0.71). All three fixtures suppressed identically."""
    _, sink, _ = _scan(
        tmp_path,
        {
            "both.py": _FIXTURE_BOTH_SHAPES,
            "helper.py": _FIXTURE_HELPER_WRAPPED,
            "ifelse.py": _FIXTURE_IFELSE_DOWNGRADE,
        },
        _PLAN_WITH_ENUMERATION,
    )
    assert sink == [], f"expected silence under an enumerated plan; got: {sink}"


def test_warns_when_none_escape_contradicted(tmp_path: Path) -> None:
    _, sink, _ = _scan(tmp_path, {"both.py": _FIXTURE_BOTH_SHAPES}, _PLAN_WITH_ESCAPE)
    assert any("falsified" in w for w in sink), f"expected the falsified-escape WARN; got: {sink}"


def test_no_warn_on_clean_script(tmp_path: Path) -> None:
    _, sink, _ = _scan(tmp_path, {"clean.py": _FIXTURE_CLEAN})
    assert sink == [], f"shrink-only smoke params are not triggers; got: {sink}"


def test_unparseable_script_warns_not_crashes(tmp_path: Path) -> None:
    result, sink, _ = _scan(tmp_path, {"broken.py": "def broken(:\n"})
    assert result == []
    assert len(sink) == 1 and "unparseable" in sink[0], (
        f"expected exactly one unparseable WARN; got: {sink}"
    )


def test_returns_empty_fail_list_always(tmp_path: Path) -> None:
    """The WARN-only contract: every scenario returns [] (never a FAIL)."""
    scenarios: list[tuple[dict[str, str], str | None]] = [
        ({"both.py": _FIXTURE_BOTH_SHAPES}, None),
        ({"both.py": _FIXTURE_BOTH_SHAPES}, _PLAN_WITHOUT_ENUMERATION),
        ({"both.py": _FIXTURE_BOTH_SHAPES}, _PLAN_WITH_ESCAPE),
        ({"both.py": _FIXTURE_BOTH_SHAPES}, _PLAN_WITH_ENUMERATION),
        ({"helper.py": _FIXTURE_HELPER_WRAPPED}, None),
        ({"ifelse.py": _FIXTURE_IFELSE_DOWNGRADE}, None),
        ({"ternary.py": _FIXTURE_TERNARY_SUBSTITUTION}, None),
        ({"clean.py": _FIXTURE_CLEAN}, None),
        ({"broken.py": "def broken(:\n"}, None),
    ]
    for i, (fixtures, plan_text) in enumerate(scenarios):
        sub = tmp_path / f"scenario_{i}"
        sub.mkdir()
        result, _, _ = _scan(sub, fixtures, plan_text)
        assert result == [], f"scenario {i} returned a FAIL list: {result}"


# --------------------------------------------------------------------------
# Review-lens surface pin (seven surfaces)
# --------------------------------------------------------------------------


def _write_lens_corpus(root: Path, *, drop: str | None = None) -> Path:
    """Build a minimal seven-surface corpus under ``root``; ``drop`` removes
    exactly one surface/token to exercise each per-surface error."""
    agents = root / ".claude" / "agents"
    rules = root / ".claude" / "rules"
    agents.mkdir(parents=True, exist_ok=True)
    rules.mkdir(parents=True, exist_ok=True)

    # (1) the rule file.
    if drop != "rule-file":
        (rules / "smoke-blind-spots.md").write_text(
            "# Smoke blind spots — enumerate what a smoke PASS does NOT certify\n",
            encoding="utf-8",
        )

    # (2) code-reviewer.md: Step 0.71 section + Blocker-tags line.
    body_tag = "" if drop == "section-body-tag" else f"tagged `{_TAG}` (SUBSTANTIVE). "
    section = (
        "### Step 0.71: Smoke blind-spot enumeration gate (any diff type)\n\n"
        f"Trigger: an unenumerated smoke-conditional branch FAILs, {body_tag}"
        f"The empty form is the literal `{_ESCAPE}`.\n\n"
    )
    if drop == "step071-section":
        section = ""
    claude_tags = "`substantive`" if drop == "claude-blocker-tag" else f"`{_TAG}`, `substantive`"
    (agents / "code-reviewer.md").write_text(
        "# code-reviewer\n\n" + section + "### Step 9: Verdict\n\n"
        f"**Blocker tags:** [{claude_tags}]\n",
        encoding="utf-8",
    )

    # (3) codex-code-reviewer.md: bullet + rubric slot + Blocker-tags line.
    bullet_tag = "." if drop == "codex-bullet-tag" else f", a single Critical tagged `{_TAG}`."
    bullet = (
        '- "Step 0.71: Smoke blind-spot enumeration gate" — unenumerated '
        f"branch FAILs{bullet_tag}\n"
        '- "Step 0.8: Read prior open binding concerns" — placeholder.\n'
    )
    if drop == "codex-heading":
        bullet = '- "Step 0.8: Read prior open binding concerns" — placeholder.\n'
    rubric = (
        "{{INLINED RUBRIC FROM code-reviewer.md Steps 0.7, 0.8}}\n"
        if drop == "codex-rubric"
        else "{{INLINED RUBRIC FROM code-reviewer.md Steps 0.7, 0.71, 0.8}}\n"
    )
    codex_tags = "`substantive`" if drop == "codex-blocker-tag" else f"`{_TAG}` | `substantive`"
    (agents / "codex-code-reviewer.md").write_text(
        "# codex-code-reviewer\n\n" + bullet + "\n" + rubric + "\n"
        f"**Blocker tags:** [{codex_tags}]\n",
        encoding="utf-8",
    )

    # (4) planner-section-reference.md: the ## 4. Design region.
    psr_escape = "" if drop == "psr-escape" else f"the literal `{_ESCAPE}`"
    (rules / "planner-section-reference.md").write_text(
        "## 4. Design\n\n"
        "- **Smoke blind-spot enumeration (REQUIRED whenever the plan "
        "declares a pre-launch smoke run).** The smoke section states what "
        f"the PASS does and does NOT certify; the empty form is {psr_escape}.\n\n"
        "## 6. Evaluation\n\nOther content.\n",
        encoding="utf-8",
    )

    # (5) critic-lens-reference.md: the Methodology lens region.
    clr_item = (
        "19. Placeholder item.\n"
        if drop == "clr-item"
        else "19. **Smoke blind-spot enumeration (any plan declaring a "
        "pre-launch smoke run).** REVISE on a missing enumeration.\n"
    )
    (rules / "critic-lens-reference.md").write_text(
        "### Methodology lens\n\n" + clr_item + "\n### Statistics & Measurement lens\n\nItems.\n",
        encoding="utf-8",
    )

    # (6) planner.md: the §4 hard-requirement capsule token.
    planner_capsule = (
        "smoke/sweep architectural parity · no all-or-nothing gates"
        if drop == "planner-capsule"
        else "smoke/sweep architectural parity · smoke blind-spot enumeration "
        "(what the PASS does NOT certify) · no all-or-nothing gates"
    )
    (agents / "planner.md").write_text("# planner\n\n" + planner_capsule + "\n", encoding="utf-8")

    # (7) critic.md: the Methodology-capsule item token.
    critic_capsule = (
        "18 persist-by-default."
        if drop == "critic-capsule"
        else "18 persist-by-default · 19 smoke blind-spot enumeration "
        "(names what the PASS does and does NOT certify)."
    )
    (agents / "critic.md").write_text("# critic\n\n" + critic_capsule + "\n", encoding="utf-8")
    return root


def test_review_lens_passes_on_complete_corpus(tmp_path: Path) -> None:
    _write_lens_corpus(tmp_path)
    errors = check_smoke_blind_spot_review_lens(repo_root=tmp_path)
    assert errors == [], f"complete corpus should pass; got: {errors}"


_DROP_CASES: list[tuple[str, str, str]] = [
    ("rule-file", "missing", "rules/smoke-blind-spots.md"),
    ("step071-section", "### Step 0.71", "agents/code-reviewer.md"),
    ("section-body-tag", "section body no longer names", "agents/code-reviewer.md"),
    ("claude-blocker-tag", "**Blocker tags:**", "agents/code-reviewer.md"),
    ("codex-heading", "copy-list token", "agents/codex-code-reviewer.md"),
    ("codex-bullet-tag", "copy-list bullet", "agents/codex-code-reviewer.md"),
    ("codex-rubric", "INLINED RUBRIC", "agents/codex-code-reviewer.md"),
    ("codex-blocker-tag", "**Blocker tags:**", "agents/codex-code-reviewer.md"),
    ("psr-escape", "## 4. Design", "rules/planner-section-reference.md"),
    ("clr-item", "Methodology lens", "rules/critic-lens-reference.md"),
    ("planner-capsule", "smoke blind-spot enumeration", "agents/planner.md"),
    ("critic-capsule", "19 smoke blind-spot enumeration", "agents/critic.md"),
]


@pytest.mark.parametrize(("drop", "token", "path_frag"), _DROP_CASES)
def test_review_lens_fails_per_missing_surface(
    tmp_path: Path, drop: str, token: str, path_frag: str
) -> None:
    _write_lens_corpus(tmp_path, drop=drop)
    errors = check_smoke_blind_spot_review_lens(repo_root=tmp_path)
    assert errors, f"drop={drop}: expected >=1 error"
    assert any(token in e and path_frag in e for e in errors), (
        f"drop={drop}: no error carries both {token!r} and {path_frag!r}; got: {errors}"
    )


def test_review_lens_passes_on_live_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binds the landed #2165 edits; the standing regression guard for
    future refactors of any of the seven surfaces."""
    monkeypatch.delenv("EPS_WORKFLOW_LINT_REPO_ROOT", raising=False)
    errors = check_smoke_blind_spot_review_lens(repo_root=None)
    assert errors == [], f"live tree should carry all seven surfaces; got: {errors}"


def test_check_smoke_blind_spot_review_lens_bundled_in_no_flags(tmp_path: Path) -> None:
    """Two-part behavioral bundling pin (the #1701 test's precedent shape).

    Part A — scoped-flag subprocess against a DRIFTED corpus (rule file
    dropped), rooted via ``EPS_WORKFLOW_LINT_REPO_ROOT``: proves the flag
    exists, the dispatch calls the function, and it emits its
    uniquely-tagged error (nonzero exit).

    Part B — no-flags OR-chain + dispatch-ladder evidence: ``main()``'s
    source names ``args.check_smoke_blind_spot_review_lens`` in BOTH the
    ``no_flags = not (...)`` OR-chain and the ``or no_flags`` dispatch
    ladder (and the WARN scanner's flag in the OR-chain, so passing it
    suppresses the default bundle).
    """
    # Part A — scoped-flag subprocess against a drifted corpus.
    _write_lens_corpus(tmp_path, drop="rule-file")
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
            "--check-smoke-blind-spot-review-lens",
            "--file",
            str(workflow_yaml_dst),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "smoke-blind-spot" in combined, (
        "smoke-blind-spot error token missing from output — the CLI flag "
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
    assert "args.check_smoke_blind_spot_review_lens" in or_chain_src, (
        "args.check_smoke_blind_spot_review_lens is NOT in the no_flags "
        "OR-chain — a bare workflow_lint.py invocation will not fire this "
        f"check. OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_smoke_blind_spots" in or_chain_src, (
        "args.check_smoke_blind_spots is NOT in the no_flags OR-chain — "
        "passing the WARN scanner flag would not suppress the default "
        f"bundle. OR-chain source:\n{or_chain_src}"
    )
    assert "args.check_smoke_blind_spot_review_lens or no_flags" in main_src, (
        "args.check_smoke_blind_spot_review_lens is NOT dispatched under "
        "`or no_flags` — the flag is defined but not bundled into the "
        "no-flags default run."
    )
