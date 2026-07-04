"""Tests for ``workflow_lint --check-jsonl-splitlines`` (#950).

The check FAILs a ``.splitlines()`` call reading JSONL content:
``json.dumps(..., ensure_ascii=False)`` leaves raw U+2028/U+2029/NEL inside
JSON strings and ``str.splitlines()`` splits on ALL Unicode line boundaries,
shredding valid records (incident #825 run-1d; eight live workflow-surface
reader sites across seven files fixed with #950).

Covers, per plan §4c: (i) each of the four signals a/b/c/d firing —
the (d) fixtures are the exact §4b-prime sibling-reader shapes plus the
round-2 ``concerns_path`` (verify_task_body.py check-14) shape; (ii) benign
non-fires (``__doc__.splitlines()``, a non-events name); (iii) the
``# JSONL_SPLITLINES_EXEMPT`` waiver in BOTH placements (same line,
preceding non-blank line); (iv) allowlist suppression; (v) the live tree
passes AND every allowlist entry matches the frozen experiment-script path
shape (never a workflow-surface file); (vi) an unparseable file is skipped
with a printed notice, never a crash or a flag; (vii) the MUTATION-VISIBLE
no-flags DISPATCH test (the ``tests/test_workflow_lint.py:3455`` pattern) —
a direct call of the check function is NOT sufficient evidence of bundling
(see the caveat at ``tests/test_workflow_lint.py:1431``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402
from workflow_lint import (  # noqa: E402
    JSONL_SPLITLINES_LEGACY_ALLOWLIST,
    check_jsonl_splitlines,
)

# The frozen allowlist path shape: legacy per-issue experiment scripts under
# scripts/ (issue*/i<digit>* prefixes) or experiment packages under
# src/explore_persona_space/experiments/ — NEVER a workflow-surface file.
ALLOWLIST_SHAPE_RE = re.compile(r"^(scripts/(issue|i\d)|src/explore_persona_space/experiments/)")


def _plant(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _run_on(monkeypatch, tmp_path: Path) -> list[str]:
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    return check_jsonl_splitlines()


# --------------------------------------------------------------------------
# (i) the four signals fire
# --------------------------------------------------------------------------


def test_signal_a_jsonl_named_read_text_chain_fires(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/offender_a.py",
        "from pathlib import Path\n"
        "def read_rows(jsonl_path: Path):\n"
        "    return [ln for ln in jsonl_path.read_text().splitlines() if ln.strip()]\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "offender_a.py:3" in errors[0], errors
    assert "jsonl-splitlines" in errors[0], errors


def test_signal_a_path_div_jsonl_literal_fires(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/offender_a2.py",
        "from pathlib import Path\n"
        "def count_rows(d: Path) -> int:\n"
        '    return len((d / "pool.jsonl").read_text().splitlines())\n',
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "offender_a2.py:3" in errors[0], errors


def test_signal_b_jsonl_named_receiver_fires(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/offender_b.py",
        "def parse(jsonl_text: str):\n    return jsonl_text.splitlines()\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "jsonl-named receiver" in errors[0], errors


def test_signal_c_jsonl_named_enclosing_function_fires(tmp_path, monkeypatch) -> None:
    # The exact pre-fix `_iter_jsonl` shape: receiver `text` carries no jsonl
    # token (read on a separate line), only the enclosing function name does —
    # this signal pins the §4b fix against reintroduction.
    _plant(
        tmp_path,
        "src/explore_persona_space/offender_c.py",
        "from pathlib import Path\n"
        "def _iter_jsonl(path: Path):\n"
        "    text = path.read_text()\n"
        "    for line in text.splitlines():\n"
        "        yield line\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "jsonl-named function" in errors[0], errors


def test_signal_d_events_path_shapes_fire(tmp_path, monkeypatch) -> None:
    # The exact §4b-prime sibling-reader shapes that evade signals (a)-(c):
    # receivers ev_path / events_path, no "jsonl" token anywhere the other
    # signals look.
    _plant(
        tmp_path,
        "scripts/offender_d.py",
        "from pathlib import Path\n"
        "def collect(ev_path: Path, events_path: Path):\n"
        "    rows = [ln for ln in ev_path.read_text().splitlines() if ln.strip()]\n"
        '    for line in events_path.read_text(errors="ignore").splitlines():\n'
        "        rows.append(line)\n"
        "    return rows\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 2, errors
    assert all("events/comments/concerns-path" in e for e in errors), errors


def test_signal_d_concerns_path_shape_fires(tmp_path, monkeypatch) -> None:
    # The exact verify_task_body.py check-14 shape fixed in #950 round 2:
    # receiver `concerns_path` carries no "jsonl" token anywhere signals
    # (a)-(c) look, so only the extended (d) base-name set catches a
    # reintroduction. Filename deliberately avoids "concerns" so the label
    # assertion cannot be satisfied by the path in the error string.
    _plant(
        tmp_path,
        "scripts/offender_d2.py",
        "from pathlib import Path\n"
        "def audit(concerns_path: Path):\n"
        "    return [ln for ln in concerns_path.read_text().splitlines() if ln.strip()]\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors
    assert "events/comments/concerns-path" in errors[0], errors


# --------------------------------------------------------------------------
# (ii) benign shapes do NOT fire
# --------------------------------------------------------------------------


def test_benign_shapes_pass(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/benign.py",
        "from pathlib import Path\n"
        "def usage() -> list[str]:\n"
        "    return (__doc__ or '').splitlines()\n"
        "def read_log(stdout_path: Path):\n"
        "    return stdout_path.read_text().splitlines()\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


# --------------------------------------------------------------------------
# (iii) the waiver comment suppresses — both placements
# --------------------------------------------------------------------------


def test_waiver_same_line_suppresses(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/waived_same_line.py",
        "from pathlib import Path\n"
        "def read_rows(jsonl_path: Path):\n"
        "    return jsonl_path.read_text().splitlines()"
        "  # JSONL_SPLITLINES_EXEMPT: ASCII-only generated file, verified safe\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_waiver_preceding_nonblank_line_suppresses(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/waived_prev_line.py",
        "from pathlib import Path\n"
        "def read_rows(jsonl_path: Path):\n"
        "    # JSONL_SPLITLINES_EXEMPT: ASCII-only generated file, verified safe\n"
        "    return jsonl_path.read_text().splitlines()\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


def test_waiver_short_reason_does_not_suppress(tmp_path, monkeypatch) -> None:
    _plant(
        tmp_path,
        "scripts/short_reason.py",
        "from pathlib import Path\n"
        "def read_rows(jsonl_path: Path):\n"
        "    return jsonl_path.read_text().splitlines()  # JSONL_SPLITLINES_EXEMPT: ok\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert len(errors) == 1, errors


# --------------------------------------------------------------------------
# (iv) allowlisted files suppress
# --------------------------------------------------------------------------


def test_allowlisted_file_suppresses(tmp_path, monkeypatch) -> None:
    # Use a REAL allowlist member's repo-relative path so the production
    # frozenset (not a test double) is exercised.
    member = "scripts/issue823_identity_baseline.py"
    assert member in JSONL_SPLITLINES_LEGACY_ALLOWLIST
    _plant(
        tmp_path,
        member,
        "from pathlib import Path\n"
        "def read_rows(jsonl_path: Path):\n"
        "    return jsonl_path.read_text().splitlines()\n",
    )
    assert _run_on(monkeypatch, tmp_path) == []


# --------------------------------------------------------------------------
# (v) live tree passes + the allowlist path-shape HARD RULE
# --------------------------------------------------------------------------


def test_live_tree_passes() -> None:
    """The real repo carries zero un-waived, un-allowlisted offenders — locks
    the allowlist to today's tree (the judge-pin test pattern). A new offender
    must be FIXED (workflow surface) or allowlisted (frozen experiment script
    only)."""
    assert check_jsonl_splitlines() == []


def test_allowlist_entries_match_experiment_script_shape() -> None:
    """HARD RULE (plan §4c): a workflow-surface file can never quietly enter
    the allowlist — every entry matches the frozen experiment-code path shape
    (per-issue scripts/ files or src/explore_persona_space/experiments/)."""
    for entry in sorted(JSONL_SPLITLINES_LEGACY_ALLOWLIST):
        assert ALLOWLIST_SHAPE_RE.match(entry), (
            f"allowlist entry {entry!r} does not match the frozen "
            f"experiment-script shape {ALLOWLIST_SHAPE_RE.pattern!r} — "
            f"workflow-surface files must be FIXED, never allowlisted"
        )


# --------------------------------------------------------------------------
# (vi) unparseable file: skip-with-report, never crash, never flag
# --------------------------------------------------------------------------


def test_unparseable_file_skipped_with_notice(tmp_path, monkeypatch, capsys) -> None:
    _plant(
        tmp_path,
        "scripts/broken.py",
        "def broken(:\n    jsonl_path.read_text().splitlines()\n",
    )
    errors = _run_on(monkeypatch, tmp_path)
    assert errors == []
    err = capsys.readouterr().err
    assert "--check-jsonl-splitlines skipped unparseable" in err, err
    assert "broken.py" in err and "SyntaxError" in err, err


# --------------------------------------------------------------------------
# (vii) the MUTATION-VISIBLE no-flags DISPATCH test (the :3455 pattern)
# --------------------------------------------------------------------------


def test_check_jsonl_splitlines_bundled_in_no_flags(tmp_path, capsys, monkeypatch) -> None:
    """The no-flags default run actually DISPATCHES the #950 check — deleting
    its ``or no_flags`` branch must fail this test (mutation-visible), closing
    the dead-tripwire gap where all direct-call tests stay green while the CLI
    never runs the check. Other bundled checks contribute unrelated errors on
    the minimal tree, so the assertion keys on the check's own diagnostic
    token + the offending path."""
    _plant(
        tmp_path,
        "scripts/foo.py",
        "from pathlib import Path\n"
        "def read_rows(jsonl_path: Path):\n"
        "    return jsonl_path.read_text().splitlines()\n",
    )
    monkeypatch.setattr(wl, "_REPO_ROOT", tmp_path)
    rc = wl.main([])
    err = capsys.readouterr().err
    assert rc != 0, f"no-flags default run exited 0 on an offending tree:\n{err}"
    assert "jsonl-splitlines" in err and "foo.py" in err, (
        f"the jsonl-splitlines diagnostic (naming foo.py) is missing from the "
        f"no-flags default run's stderr — the check is not bundled into "
        f"no_flags:\n{err}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
