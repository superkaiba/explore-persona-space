"""Tests for the #2168 JSON-guard UnicodeDecodeError reintroduction guard in
``scripts/workflow_lint.py``: ``check_json_guard_unicode``
(``--check-json-guard-unicode``, bundled into the no-flags default run).

Predicate under test (plan #2168 v2 §4-D2, the EXTENDED form the sweep
instrument ``scripts/issue2168_sweep.py`` battle-tested over 186 live units):
flag a unit — an ``ast.Try``, an ``ast.TryStar`` (``except*``), or a
``with contextlib.suppress(...)`` item — iff its caught/suppressed-name union
contains BOTH a ``JSONDecodeError``-named exception AND an OSError-family
name, contains NO safe name (UnicodeDecodeError / ValueError / Exception /
BaseException), and carries no ``# JSON_GUARD_UNICODE_EXEMPT: <reason>``
waiver.

Coverage map:

1.  13 positive fixtures (parametrized, all flagged): canonical order;
    reversed order; wider tuple; split-handler; ``from json import
    JSONDecodeError`` alias; ``json.load(open(...))`` body; two-arg
    ``contextlib.suppress`` (attribute form); ``import contextlib as ctx``
    aliased attribute form (round 3, concern
    ``json-guard-custom-suppress-union-fp``: the attribute-base tightening
    must still resolve import aliases); bare-name ``suppress``
    (from-import form); ``except*`` group; split-suppress within ONE
    ``with`` statement (round 2, concern
    ``json-guard-split-suppress-undisclosed-miss``); nested-literal-tuple
    handler (round 2 Minor — TypeError-at-match-time on Python 3, flagged
    as the banned shape in intent); literal-tuple suppress arg (round 2 —
    semantically live, ``issubclass`` recurses).
2.  8 negative fixtures (parametrized, none flagged): three-element fixed
    try form; three-arg fixed suppress; bare ``ValueError``;
    ``except Exception``; safe-member tuple; adequate waiver (line above);
    mixed real/custom suppress in ONE ``with`` (round 3, concern
    ``json-guard-custom-suppress-union-fp`` — an unrelated custom
    ``.suppress`` context manager must not join the union); safe-name
    sibling suppress item (round 3 — pins the round-2 FP-removal the r2
    marker's "strictly wider" claim mis-described: a statement whose
    sibling suppress item carries a safe name is correctly un-flagged).
3.  Waiver with a too-short reason -> still flagged; same-line waiver on the
    handler line -> accepted; waiver separated by a blank line -> accepted
    (round 2, concern ``waiver-placement-parity``: pins the house
    backward-walk-over-blanks convention as DELIBERATE — the disposable
    sweep's exact-previous-line form is the one-off simplification).
4.  Documented-miss fixtures: ``from contextlib import suppress as quiet``
    is NOT flagged (#2168 plan v2 §4-D5b); NESTED ``with`` statements
    each suppressing one half are NOT flagged (round 2 disclosure); and
    ``suppress`` reached through a NON-contextlib attribute base (a
    re-export ``helpers.suppress``) is NOT flagged (round 3 direction
    tradeoff of the attribute-base tightening) — each pins a disclosed
    false negative as deliberate behavior, so a future predicate change
    that starts covering it consciously updates the disclosure.
5.  FORM-SPECIFIC messages (#2168 plan v2 Must-Fix 1c): a try unit's message
    carries the TUPLE fix and never mentions suppress; a suppress unit's
    message carries the SUPPRESS-ARGS fix and never the tuple-form fix.
6.  Recursive-enumeration self-check (nested subdirectory fixture) plus a
    non-empty-enumeration assert inside the ``_scan`` helper (the
    ``test_cap_env_read_is_single_sourced`` broken-glob-fails-loud model).
7.  Live-tree PASS: ``check_json_guard_unicode()`` over the real repo
    returns ``[]`` (the post-sweep drift pin).
8.  ``test_check_json_guard_unicode_bundled_in_no_flags`` — the NON-VACUOUS
    no-flags bundling source pin (the #1385/#1233
    ``test_pipe_python_bundled_in_no_flags_source_pin`` shape): a later
    dispatch refactor cannot silently unbundle the check (#1385 v1 /
    #1648 v2 shipped exactly that regression).
9.  Sweep CLI mixed-mode pin (round 3, concern
    ``sweep-check-readonly-mixed-mode``): argparse REJECTS
    ``--check --apply`` (mutually exclusive mode group, exit 2), and
    ``--check`` reports read-only — the ``apply_edits`` mutation seam is
    stubbed to raise and must never be invoked.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2168_sweep  # noqa: E402
from workflow_lint import (  # noqa: E402
    JSON_GUARD_UNICODE_WAIVER_MIN_REASON_CHARS,
    check_json_guard_unicode,
)

_LINT = _SCRIPTS / "workflow_lint.py"


def _scan(tmp_path: Path, source: str) -> list[str]:
    """Write ``source`` into a tmp ``scripts/`` tree and run the check on it
    alone via the ``roots`` override. Self-check: the enumeration must be
    non-empty so a silently-broken glob fails loud, never passes vacuously."""
    root = tmp_path / "scripts"
    root.mkdir(parents=True, exist_ok=True)
    (root / "fixture.py").write_text(source, encoding="utf-8")
    assert sorted(root.rglob("*.py")), "fixture enumeration is empty — the glob is broken"
    return check_json_guard_unicode(roots=(root,))


# ---------------------------------------------------------------------------
# Positive fixtures — every shape the #2168 sweep found live (or covered
# pre-emptively), all flagged. The scripts are only ast.parse'd, never
# executed, so the bodies are inert by construction.
# ---------------------------------------------------------------------------

_POSITIVE_FIXTURES: dict[str, str] = {
    "canonical-order": """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
""",
    "reversed-order": """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
""",
    "wider-tuple": """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
""",
    "split-handler": """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    except OSError:
        return None
""",
    "from-import-alias": """\
from json import JSONDecodeError, loads
from pathlib import Path


def load(p: Path):
    try:
        return loads(p.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError):
        return None
""",
    "json-load-open": """\
import json


def load(path: str):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
""",
    "suppress-attribute": """\
import contextlib
import json
from pathlib import Path


def load(p: Path):
    data = None
    with contextlib.suppress(json.JSONDecodeError, OSError):
        data = json.loads(p.read_text(encoding="utf-8"))
    return data
""",
    "suppress-bare-name": """\
import json
from contextlib import suppress
from pathlib import Path


def load(p: Path):
    data = None
    with suppress(json.JSONDecodeError, OSError):
        data = json.loads(p.read_text(encoding="utf-8"))
    return data
""",
    # Round 3 (concern json-guard-custom-suppress-union-fp): the attribute
    # form is tightened to contextlib import bindings — an
    # `import contextlib as ctx` alias must STILL be resolved and flagged.
    "suppress-import-alias": """\
import contextlib as ctx
import json
from pathlib import Path


def load(p: Path):
    data = None
    with ctx.suppress(json.JSONDecodeError, OSError):
        data = json.loads(p.read_text(encoding="utf-8"))
    return data
""",
    "except-star-group": """\
import json
from pathlib import Path


def load(p: Path):
    data = None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except* (json.JSONDecodeError, OSError):
        pass
    return data
""",
    # Round 2 (concern json-guard-split-suppress-undisclosed-miss): the
    # split-suppress form WITHIN ONE with statement — two suppress items
    # whose UNION trips the predicate; probe-confirmed 0 findings pre-fix.
    "split-suppress-one-with": """\
import contextlib
import json
from pathlib import Path


def load(p: Path):
    data = None
    with contextlib.suppress(json.JSONDecodeError), contextlib.suppress(OSError):
        data = json.loads(p.read_text(encoding="utf-8"))
    return data
""",
    # Round 2 (folded Minor): NESTED literal tuple in the handler type.
    # On Python 3 this raises TypeError at match time instead of catching
    # (probe-verified 3.12) — still flagged: it is the banned guard shape
    # in intent and the message's flat-tuple fix repairs both defects.
    "nested-tuple-handler": """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ((json.JSONDecodeError, OSError),):
        return None
""",
    # Round 2: a literal-tuple SUPPRESS arg is semantically live
    # (issubclass recurses into nested tuples; probe-verified 3.12).
    "suppress-tuple-arg": """\
import contextlib
import json
from pathlib import Path


def load(p: Path):
    data = None
    with contextlib.suppress((json.JSONDecodeError, OSError)):
        data = json.loads(p.read_text(encoding="utf-8"))
    return data
""",
}


@pytest.mark.parametrize("name", sorted(_POSITIVE_FIXTURES))
def test_positive_fixture_is_flagged(tmp_path: Path, name: str) -> None:
    """Each banned shape yields exactly one finding naming the fixture."""
    errors = _scan(tmp_path, _POSITIVE_FIXTURES[name])
    assert len(errors) == 1, f"{name}: expected exactly 1 finding, got: {errors}"
    assert "fixture.py" in errors[0]
    assert "json-guard-unicode" in errors[0]


# ---------------------------------------------------------------------------
# Negative fixtures — safe shapes, none flagged.
# ---------------------------------------------------------------------------

_NEGATIVE_FIXTURES: dict[str, str] = {
    "fixed-three-element-try": """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
""",
    "fixed-three-arg-suppress": """\
import contextlib
import json
from pathlib import Path


def load(p: Path):
    data = None
    with contextlib.suppress(json.JSONDecodeError, OSError, UnicodeDecodeError):
        data = json.loads(p.read_text(encoding="utf-8"))
    return data
""",
    "bare-valueerror": """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
""",
    "except-exception": """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
""",
    "safe-member-in-tuple": """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return None
""",
    "waiver-adequate-line-above": """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    # JSON_GUARD_UNICODE_EXEMPT: bytes are pre-validated as UTF-8 upstream
    except (json.JSONDecodeError, OSError):
        return None
""",
    # Round 3 (concern json-guard-custom-suppress-union-fp): `custom.suppress`
    # is an UNRELATED context manager (its base is not a contextlib import
    # binding), so it must NOT join the per-statement union — probe-confirmed
    # FLAGGED pre-fix (the old any-`.suppress` breadth), clean post-fix.
    "mixed-real-and-custom-suppress": """\
import contextlib
import json
from pathlib import Path


def load(p: Path, custom):
    data = None
    with contextlib.suppress(json.JSONDecodeError), custom.suppress(OSError):
        data = json.loads(p.read_text(encoding="utf-8"))
    return data
""",
    # Round 3 (r2 marker correction): a SIBLING suppress item carrying a safe
    # name un-flags the statement — ValueError covers UnicodeDecodeError at
    # runtime, so the statement genuinely suppresses it and the round-2
    # per-statement union correctly reads it as already-total. Old per-item
    # behavior FLAGGED item 1 (a semantic false positive); this pins the
    # FP-removal the r2 marker's "strictly wider" claim mis-described.
    "safe-sibling-suppress-item": """\
import contextlib
import json
from pathlib import Path


def load(p: Path):
    data = None
    with contextlib.suppress(json.JSONDecodeError, OSError), contextlib.suppress(ValueError):
        data = json.loads(p.read_text(encoding="utf-8"))
    return data
""",
}


@pytest.mark.parametrize("name", sorted(_NEGATIVE_FIXTURES))
def test_negative_fixture_is_not_flagged(tmp_path: Path, name: str) -> None:
    """Safe shapes and adequately-waived units produce no findings."""
    errors = _scan(tmp_path, _NEGATIVE_FIXTURES[name])
    assert errors == [], f"{name}: expected no findings, got: {errors}"


def test_waiver_with_short_reason_still_flagged(tmp_path: Path) -> None:
    """A waiver token with a reason below the minimum is NOT honored."""
    short = "ok"
    assert len(short) < JSON_GUARD_UNICODE_WAIVER_MIN_REASON_CHARS
    source = f"""\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    # JSON_GUARD_UNICODE_EXEMPT: {short}
    except (json.JSONDecodeError, OSError):
        return None
"""
    errors = _scan(tmp_path, source)
    assert len(errors) == 1, f"short-reason waiver must not suppress: {errors}"


def test_waiver_on_flagged_line_accepted(tmp_path: Path) -> None:
    """The waiver is honored on the flagged handler line itself too."""
    source = """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # JSON_GUARD_UNICODE_EXEMPT: pre-decoded bytes
        return None
"""
    assert _scan(tmp_path, source) == []


def test_documented_miss_aliased_suppress_not_flagged(tmp_path: Path) -> None:
    """PINS the disclosed false negative: ``from contextlib import suppress
    as quiet`` evades the name-based match (0 live instances at plan time).
    A future predicate change that starts covering the aliased form must
    consciously update the check docstring's disclosure list — this test
    turning red is that signal, not a defect."""
    source = """\
import json
from contextlib import suppress as quiet
from pathlib import Path


def load(p: Path):
    data = None
    with quiet(json.JSONDecodeError, OSError):
        data = json.loads(p.read_text(encoding="utf-8"))
    return data
"""
    assert _scan(tmp_path, source) == []


def test_documented_miss_nested_with_suppress_not_flagged(tmp_path: Path) -> None:
    """PINS the round-2 disclosed false negative (concern
    ``json-guard-split-suppress-undisclosed-miss``, residual): NESTED
    ``with`` statements each suppressing one half are SEPARATE ``With``
    nodes, so the per-STATEMENT suppress union — which closes the
    split-suppress form within one ``with`` (the round-2 fix) — never sees
    the combined name set. 0 live instances. A future predicate change that
    starts covering the nested form must consciously update the check
    docstring's disclosure list — this test turning red is that signal,
    not a defect."""
    source = """\
import contextlib
import json
from pathlib import Path


def load(p: Path):
    data = None
    with contextlib.suppress(json.JSONDecodeError):
        with contextlib.suppress(OSError):
            data = json.loads(p.read_text(encoding="utf-8"))
    return data
"""
    assert _scan(tmp_path, source) == []


def test_documented_miss_nonimport_attribute_suppress_not_flagged(tmp_path: Path) -> None:
    """PINS the round-3 disclosed false negative (concern
    ``json-guard-custom-suppress-union-fp``, the tightening's direction
    tradeoff): ``suppress`` reached through a NON-contextlib attribute
    base — here a re-export ``helpers.suppress`` — is missed, because the
    attribute form matches only when its base Name is a contextlib import
    binding (the tightening that stops unrelated custom ``.suppress``
    context managers joining the per-statement union as false positives;
    the old any-``.suppress`` breadth DID catch this shape). Requires
    deliberate binding indirection; 0 live instances at plan time. A future
    predicate change that starts covering it must consciously update the
    check docstring's disclosure list — this test turning red is that
    signal, not a defect."""
    source = """\
import json
from pathlib import Path

from myproject import helpers


def load(p: Path):
    data = None
    with helpers.suppress(json.JSONDecodeError, OSError):
        data = json.loads(p.read_text(encoding="utf-8"))
    return data
"""
    assert _scan(tmp_path, source) == []


def test_waiver_honored_across_blank_gap(tmp_path: Path) -> None:
    """DELIBERATE-divergence pin (round 2, concern
    ``waiver-placement-parity``): the lint's waiver walk skips blank lines
    back to the preceding NON-BLANK line — the house convention,
    byte-parallel with ``_jsonl_splitlines_waiver_present`` — so a waiver
    separated from the flagged line by blank lines IS honored. The
    disposable sweep instrument (``scripts/issue2168_sweep.py``) checks
    only the exact previous physical line; that divergence is deliberate
    (the lint is the durable, binding surface and is STRICTER where it
    matters — the >=10-char reason floor), and this test pins the
    blank-gap behavior as intended, not accidental."""
    source = """\
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    # JSON_GUARD_UNICODE_EXEMPT: bytes are pre-validated as UTF-8 upstream

    except (json.JSONDecodeError, OSError):
        return None
"""
    assert _scan(tmp_path, source) == []


def test_try_message_is_tuple_form_specific(tmp_path: Path) -> None:
    """A flagged try unit is told to extend the TUPLE, and its message never
    mentions the suppress form."""
    [error] = _scan(tmp_path, _POSITIVE_FIXTURES["canonical-order"])
    assert "except (json.JSONDecodeError, OSError, UnicodeDecodeError)" in error
    assert "suppress" not in error


def test_suppress_message_is_args_form_specific(tmp_path: Path) -> None:
    """A flagged suppress unit is told to extend the SUPPRESS ARGS and is
    NEVER shown the tuple form — printing the tuple fix at a suppress site
    would teach the exact rewrite this check also flags (#2168 plan v2
    Must-Fix 1c)."""
    [error] = _scan(tmp_path, _POSITIVE_FIXTURES["suppress-attribute"])
    assert "contextlib.suppress(json.JSONDecodeError, OSError, UnicodeDecodeError)" in error
    assert "except (" not in error


def test_except_star_message_names_except_star(tmp_path: Path) -> None:
    """An ``except*`` unit's message shows the ``except*`` tuple fix."""
    [error] = _scan(tmp_path, _POSITIVE_FIXTURES["except-star-group"])
    assert "except* (json.JSONDecodeError, OSError, UnicodeDecodeError)" in error


def test_recursive_enumeration_scans_subdirectories(tmp_path: Path) -> None:
    """The walk is recursive: an offender in a scripts/ SUBDIRECTORY is
    found (scripts/ has .py subdirectories on the live tree — a
    non-recursive glob would silently skip them)."""
    root = tmp_path / "scripts"
    nested = root / "issue_9999"
    nested.mkdir(parents=True)
    (root / "clean.py").write_text("X = 1\n", encoding="utf-8")
    (nested / "offender.py").write_text(_POSITIVE_FIXTURES["canonical-order"], encoding="utf-8")
    errors = check_json_guard_unicode(roots=(root,))
    assert len(errors) == 1 and "offender.py" in errors[0], errors


def test_multiple_units_in_one_file_all_flagged(tmp_path: Path) -> None:
    """Two independent offending units in one module yield two findings."""
    source = (
        _POSITIVE_FIXTURES["canonical-order"]
        + "\n\n"
        + _POSITIVE_FIXTURES["suppress-attribute"].replace("def load(", "def load2(")
    )
    errors = _scan(tmp_path, source)
    assert len(errors) == 2, errors


def test_live_tree_passes() -> None:
    """The post-sweep drift pin: the real scripts/ + src/ trees carry ZERO
    unwaived (JSONDecodeError, OSError-family)-without-UnicodeDecodeError
    guard units (#2168 swept all 186; a red run here means a fresh offender
    landed — fix it or waive it, never loosen the predicate)."""
    errors = check_json_guard_unicode()
    assert errors == [], (
        "scripts/ + src/ carry unwaived JSON-guard units missing "
        "UnicodeDecodeError (#2164/#2168 crash class):\n" + "\n".join(errors)
    )


def test_check_json_guard_unicode_bundled_in_no_flags() -> None:
    """NON-VACUOUS no-flags bundling pin (the #1385/#1233
    ``test_pipe_python_bundled_in_no_flags_source_pin`` shape):
    ``check_json_guard_unicode`` must be dispatched by the BARE
    ``workflow_lint.py`` run — the Step 9c gate and the inline payload lint
    gate both invoke the no-flags run, so absence from the ``no_flags``
    disjunction turns the guard into an opt-in (#1385 v1 / #1648 v2 shipped
    exactly that silent-unbundling regression). Source-inspection assert on
    the dispatch branch + the no_flags detection-tuple membership (an
    exit-0-on-clean-tree assert is vacuous — a clean tree exits 0 whether or
    not the check is dispatched)."""
    src = _LINT.read_text(encoding="utf-8")
    assert re.search(
        r"if args\.check_json_guard_unicode or no_flags:\s*\n"
        r"\s*errors\.extend\(check_json_guard_unicode\(\)\)",
        src,
    ), "check_json_guard_unicode is not dispatched on the no-flags branch"
    assert "or args.check_json_guard_unicode" in src, (
        "--check-json-guard-unicode is missing from the no_flags detection tuple"
    )


# ---------------------------------------------------------------------------
# Sweep CLI mixed-mode pins (round 3, concern sweep-check-readonly-mixed-mode):
# the disposable sweep instrument's --check mode is advertised read-only, so
# (a) argparse must REJECT combining it with the mutating --apply, and
# (b) --check must report without ever reaching the apply_edits mutation seam.
# ---------------------------------------------------------------------------


def test_sweep_check_apply_rejected_by_argparse(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check --apply`` is rejected at parse time (mutually exclusive mode
    group, argparse usage-error exit 2) — a read-only flag that silently
    combines with the mutating mode is the false label this pins against.
    The scan/apply seams are stubbed to raise so a regression that drops the
    exclusivity group fails HERE, legibly, instead of scanning (or mutating)
    the live tree."""

    def _forbidden_scan(repo_root: Path) -> list[issue2168_sweep.Finding]:
        raise AssertionError("scan must not run when argparse rejects the argv")

    def _forbidden_apply(findings: list[issue2168_sweep.Finding]) -> list[Path]:
        raise AssertionError("apply_edits must not run when argparse rejects the argv")

    monkeypatch.setattr(issue2168_sweep, "scan", _forbidden_scan)
    monkeypatch.setattr(issue2168_sweep, "apply_edits", _forbidden_apply)
    monkeypatch.setattr(sys, "argv", ["issue2168_sweep.py", "--check", "--apply"])
    with pytest.raises(SystemExit) as excinfo:
        issue2168_sweep.main()
    assert excinfo.value.code == 2, "argparse usage errors exit 2"
    assert "not allowed with argument" in capsys.readouterr().err


def test_sweep_check_reports_without_mutation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check`` alone dispatches the report path read-only: findings are
    printed with the TOTAL trailer, the exit code is 1 iff findings exist,
    and the ``apply_edits`` mutation seam (stubbed to raise) is never
    invoked. The scan seam is stubbed at the filesystem boundary with a
    signature-conformant fake returning one synthetic Finding, so the test
    is deterministic on any tree state."""
    synthetic = issue2168_sweep.Finding(
        path=issue2168_sweep.REPO_ROOT / "scripts" / "synthetic_fixture.py",
        lineno=7,
        form="try",
        names={"JSONDecodeError", "OSError"},
        insert_line=7,
        insert_col=40,
    )

    def _fake_scan(repo_root: Path) -> list[issue2168_sweep.Finding]:
        assert repo_root == issue2168_sweep.REPO_ROOT
        return [synthetic]

    def _forbidden_apply(findings: list[issue2168_sweep.Finding]) -> list[Path]:
        raise AssertionError("apply_edits must never be invoked under --check")

    monkeypatch.setattr(issue2168_sweep, "scan", _fake_scan)
    monkeypatch.setattr(issue2168_sweep, "apply_edits", _forbidden_apply)
    monkeypatch.setattr(sys, "argv", ["issue2168_sweep.py", "--check"])
    rc = issue2168_sweep.main()
    out = capsys.readouterr().out
    assert "scripts/synthetic_fixture.py:7: [try]" in out
    assert "TOTAL: 1 units / 1 files" in out
    assert rc == 1, "findings exist and no edit was applied -> exit 1"
