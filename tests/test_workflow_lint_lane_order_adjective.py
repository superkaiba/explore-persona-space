"""Tests for ``--check-lane-order-adjective`` (#2298 Part 3/4).

The check FAILs stale auto-lane-order prose on the prescriptive surface
(.claude/agents, .claude/skills SKILL.md + issue step companions,
.claude/rules, CLAUDE.md): family 1 = a ``<lane>-first`` / ``<lane> first``
adjective naming a NON-head lane with order vocabulary within +/-1 physical
line; family 2 = a ``DEFAULT_AUTO_LANE_ORDER = ("<head>", ...)``
transcription whose head differs from the live one. The head is AST-read
from router.py SOURCE (never hardcoded); unresolvable heads SKIP loud.
Plan #2298 Part 4's numbering, in order:

- T1  stale verbatim line (the pre-fix critic-lens-reference.md:410 text)
      => exactly 1 finding naming file, line, and the resolved head; the
      single-flag CLI exits nonzero (FAIL posture).
- T2  corrected line => 0 findings; single-flag CLI exits 0.
- T3  head-driven, not hardcoded: a fellows-head fixture router flips the
      verdict (fellows-first passes, runpod-first fails).
- T4  waiver honored (same line / preceding window / top-of-file banner);
      a sub-10-char reason does NOT waive.
- T5  fail-open SKIPs: missing router, unparseable router, absent
      ``_default_auto_lane_order``, disagreeing branches, and the
      MIXED-return fixture (one tuple branch + one ``return SOME_CONST``)
      => ``skipped`` True + non-empty ``skip_reason`` + 0 findings, never
      a head resolved from the tuple branch alone.
- T6  excluded-scope: the same stale line under ``.claude/agent-memory/``
      and ``docs/methodology/`` => 0 findings (historical surfaces).
- T7  armed-not-inert against the LIVE tree: ``skipped`` False, a resolved
      head, ``files_scanned > 0``, 0 findings post-#2298.
- T8  the ``_gcp_pin_excluded`` self-exclusion trap (#2018): a repo root
      that ITSELF sits under ``.claude/worktrees/`` still scans its own
      files (root-RELATIVE exclusion, not absolute-path match).
- T9  no-flags BUNDLING pin (the house
      ``test_no_flags_default_run_pins_failure_lesson_field_contract``
      pattern): ``main([])`` over a stale fixture tree emits this check's
      own finding token — a registered-but-never-dispatched check fails
      here (#1385).
- T10 compound guard: ``gcp-first-resort`` and ``fellows firstly`` in an
      order-context line => 0 findings (the trailing ``(?![\\w-])`` guard).
- T11 order-context window: ``check fellows first before runpod`` with no
      order vocabulary within +/-1 line => 0 findings; the same adjective
      WITH ``auto default`` on the line => 1 finding.
- T12 trigger family 2: a ``DEFAULT_AUTO_LANE_ORDER = ("fellows", ...)``
      transcription against a runpod-head router => 1 finding (incl. the
      hard-wrapped form); a runpod-head transcription => 0 findings.
- T13 the window's +/-1-line GRAIN, pinned (v4 MF1): vocabulary on N-1 and
      on N+1 each => 1 finding; vocabulary at distance 2 => 0 findings —
      a future narrowing to same-line fails this test instead of silently
      dropping half the real hits (2 of the 4 live 2026-08 hits were
      hard-wrapped).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import workflow_lint as wl  # noqa: E402

_ROUTER_REL = "src/explore_persona_space/backends/router.py"

_ROUTER_RUNPOD = (
    "GCP_PROVISIONING_DISABLED: bool = True\n"
    'DEFAULT_FREE_LANE_ORDER = ("nibi", "fir", "mila")\n'
    "\n"
    "def _default_auto_lane_order():\n"
    "    if GCP_PROVISIONING_DISABLED:\n"
    '        return ("runpod", "fellows", *DEFAULT_FREE_LANE_ORDER)\n'
    '    return ("runpod", "fellows", "gcp", *DEFAULT_FREE_LANE_ORDER)\n'
    "\n"
    "DEFAULT_AUTO_LANE_ORDER = _default_auto_lane_order()\n"
)

_ROUTER_FELLOWS = _ROUTER_RUNPOD.replace('"runpod", "fellows"', '"fellows", "runpod"')

# The verbatim pre-fix critic-lens-reference.md:410 text (the incident line).
_STALE_LINE = (
    "    backend router will ACTUALLY provision — under the standing fellows-first `auto` default\n"
)
_CORRECTED_LINE = (
    "    backend router will ACTUALLY provision — under the standing runpod-first `auto` default\n"
)


def _plant(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _mk_repo(tmp_path: Path, *, router_text: str | None = _ROUTER_RUNPOD) -> Path:
    if router_text is not None:
        _plant(tmp_path, _ROUTER_REL, router_text)
    return tmp_path


def _report(root: Path) -> dict[str, object]:
    return wl.lane_order_adjective_report(repo_root=root)


# --------------------------------------------------------------------------
# T1 / T2: the incident line, stale vs corrected, + FAIL-posture CLI rc
# --------------------------------------------------------------------------


def test_t1_stale_verbatim_line_one_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _mk_repo(tmp_path)
    _plant(root, ".claude/rules/stale.md", "# heading\n\n" + _STALE_LINE)
    report = _report(root)
    findings = report["findings"]
    assert report["skipped"] is False and report["head"] == "runpod"
    assert isinstance(findings, list) and len(findings) == 1, findings
    assert "stale.md:3" in findings[0] and "'runpod'" in findings[0], findings[0]
    assert "[family-1]" in findings[0], findings[0]
    # FAIL posture: the single-flag CLI exits nonzero on the stale fixture.
    monkeypatch.setattr(wl, "_REPO_ROOT", root)
    rc = wl.main(["--check-lane-order-adjective"])
    capsys.readouterr()
    assert rc != 0


def test_t2_corrected_line_zero_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _mk_repo(tmp_path)
    _plant(root, ".claude/rules/fixed.md", "# heading\n\n" + _CORRECTED_LINE)
    report = _report(root)
    assert report["skipped"] is False and report["findings"] == []
    assert report["files_scanned"] > 0
    monkeypatch.setattr(wl, "_REPO_ROOT", root)
    rc = wl.main(["--check-lane-order-adjective"])
    capsys.readouterr()
    assert rc == 0


# --------------------------------------------------------------------------
# T3: head-driven, never hardcoded (the K2 pin)
# --------------------------------------------------------------------------


def test_t3_head_driven_not_hardcoded(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path, router_text=_ROUTER_FELLOWS)
    _plant(root, ".claude/rules/a.md", _STALE_LINE)  # fellows-first: now the HEAD
    _plant(root, ".claude/rules/b.md", _CORRECTED_LINE)  # runpod-first: now stale
    report = _report(root)
    assert report["head"] == "fellows"
    findings = report["findings"]
    assert isinstance(findings, list) and len(findings) == 1, findings
    assert "b.md" in findings[0] and "'fellows'" in findings[0], findings[0]


# --------------------------------------------------------------------------
# T4: waiver placements + the >=10-char reason floor
# --------------------------------------------------------------------------


def test_t4_waiver_honored_and_short_reason_rejected(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    waiver = "<!-- LANE-ORDER-HISTORICAL: pre-#2054 era record kept verbatim -->"
    # (a) same line
    _plant(root, ".claude/rules/w_same.md", _STALE_LINE.rstrip("\n") + f" {waiver}\n")
    # (b) preceding window (waiver 3 lines above the hit)
    _plant(root, ".claude/rules/w_window.md", f"{waiver}\n\n\n{_STALE_LINE}")
    # (c) top-of-file banner: waiver at line 1, hit at line 60 (outside the
    #     preceding-40 window from the hit, inside the first-40-lines scope)
    _plant(root, ".claude/rules/w_banner.md", waiver + "\n" + "filler\n" * 58 + _STALE_LINE)
    report = _report(root)
    assert report["findings"] == [], report["findings"]
    # A sub-10-char reason does NOT waive.
    root2 = _mk_repo(tmp_path / "short", router_text=_ROUTER_RUNPOD)
    _plant(
        root2,
        ".claude/rules/w_short.md",
        _STALE_LINE.rstrip("\n") + " <!-- LANE-ORDER-HISTORICAL: old -->\n",
    )
    report2 = _report(root2)
    findings2 = report2["findings"]
    assert isinstance(findings2, list) and len(findings2) == 1, findings2


# --------------------------------------------------------------------------
# T5: fail-open SKIPs (incl. the v4 mixed-return fixture)
# --------------------------------------------------------------------------


def test_t5_fail_open_skips(tmp_path: Path) -> None:
    cases: list[tuple[str, str | None, str]] = [
        ("missing", None, "router-unreadable"),
        ("syntax", "def _default_auto_lane_order(:\n", "head-unresolved"),
        ("absent_fn", "X = 1\n", "head-unresolved"),
        (
            "disagree",
            "def _default_auto_lane_order():\n"
            "    if X:\n"
            '        return ("runpod", "fellows")\n'
            '    return ("fellows", "runpod")\n',
            "head-unresolved",
        ),
        (
            # v4 SF4: one tuple branch + one non-tuple return => skipped,
            # NEVER a head resolved from the tuple branch alone.
            "mixed_return",
            'SOME_CONST = ("runpod",)\n'
            "def _default_auto_lane_order():\n"
            "    if X:\n"
            '        return ("runpod", "fellows")\n'
            "    return SOME_CONST\n",
            "head-unresolved",
        ),
    ]
    for name, router_text, expected_reason in cases:
        root = _mk_repo(tmp_path / name, router_text=router_text)
        _plant(root, ".claude/rules/stale.md", _STALE_LINE)
        report = _report(root)
        assert report["skipped"] is True, (name, report)
        assert report["skip_reason"] == expected_reason, (name, report["skip_reason"])
        assert report["head"] is None and report["findings"] == [], (name, report)


# --------------------------------------------------------------------------
# T6: historical surfaces are out of scope
# --------------------------------------------------------------------------


def test_t6_excluded_scope_zero_findings(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    _plant(root, ".claude/agent-memory/implementer/mem.md", _STALE_LINE)
    _plant(root, "docs/methodology/issue_1.md", _STALE_LINE)
    report = _report(root)
    assert report["findings"] == [], report["findings"]
    scanned = report["scanned_files"]
    assert isinstance(scanned, list)
    assert not any("agent-memory" in s or s.startswith("docs/") for s in scanned), scanned


# --------------------------------------------------------------------------
# T7: armed-not-inert against the LIVE tree (the A6 pin)
# --------------------------------------------------------------------------


def test_t7_live_tree_armed_not_inert() -> None:
    report = wl.lane_order_adjective_report(repo_root=_REPO_ROOT)
    assert report["skipped"] is False
    assert report["head"] is not None
    files_scanned = report["files_scanned"]
    assert isinstance(files_scanned, int) and files_scanned > 0
    assert report["findings"] == [], report["findings"]


# --------------------------------------------------------------------------
# T8: repo root itself under .claude/worktrees/ (the #2018 trap)
# --------------------------------------------------------------------------


def test_t8_worktree_rooted_repo_still_scans(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path / ".claude" / "worktrees" / "issue-9999")
    _plant(root, ".claude/rules/stale.md", _STALE_LINE)
    report = _report(root)
    findings = report["findings"]
    assert report["skipped"] is False
    assert isinstance(findings, list) and len(findings) == 1, findings


# --------------------------------------------------------------------------
# T9: the no-flags BUNDLING pin (v3 MF2; the #1385 silent-disablement class)
# --------------------------------------------------------------------------


def test_check_lane_order_adjective_bundled_in_no_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The no-flags default run actually DISPATCHES the check — deleting its
    ``or no_flags`` branch must fail this test. Other bundled checks
    contribute unrelated errors on the minimal tree, so the assertion keys
    ONLY on this check's own finding token."""
    root = _mk_repo(tmp_path)
    _plant(root, ".claude/rules/stale.md", _STALE_LINE)
    monkeypatch.setattr(wl, "_REPO_ROOT", root)
    wl.main([])
    err = capsys.readouterr().err
    assert "stale lane-order adjective" in err and "stale.md" in err, (
        f"the lane-order finding (naming stale.md) is missing from the no-flags "
        f"default run's stderr — the check is not bundled into no_flags:\n{err}"
    )


# --------------------------------------------------------------------------
# T10: the trailing (?![\w-]) compound guard (v3 SF5)
# --------------------------------------------------------------------------


def test_t10_compound_guard_zero_findings(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    _plant(
        root,
        ".claude/rules/compound.md",
        "the auto default honors the gcp-first-resort convention\n"
        "fellows firstly, the chain default applies\n",
    )
    report = _report(root)
    assert report["findings"] == [], report["findings"]


# --------------------------------------------------------------------------
# T11: the order-context vocabulary window (v3 SF5)
# --------------------------------------------------------------------------


def test_t11_order_context_window(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    _plant(
        root,
        ".claude/rules/natural.md",
        "Some sentence with no vocabulary.\n"
        "check fellows first before runpod\n"
        "Another plain sentence.\n",
    )
    _plant(root, ".claude/rules/vocab.md", "the fellows-first auto default\n")
    report = _report(root)
    findings = report["findings"]
    assert isinstance(findings, list) and len(findings) == 1, findings
    assert "vocab.md" in findings[0], findings[0]


# --------------------------------------------------------------------------
# T12: trigger family 2 — the transcription head (v3 SF3)
# --------------------------------------------------------------------------


def test_t12_family2_transcription(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    _plant(
        root,
        ".claude/rules/bad_transcription.md",
        'the order is `DEFAULT_AUTO_LANE_ORDER = ("fellows", "runpod", "nibi")` today\n',
    )
    _plant(
        root,
        ".claude/rules/bad_wrapped.md",
        # Hard-wrapped transcription: family 2 scans the whole text.
        'DEFAULT_AUTO_LANE_ORDER = (\n    "fellows", "runpod")\n',
    )
    _plant(
        root,
        ".claude/rules/good_transcription.md",
        '`DEFAULT_AUTO_LANE_ORDER = ("runpod", "fellows", "nibi", "fir", "mila")`\n',
    )
    report = _report(root)
    findings = report["findings"]
    assert isinstance(findings, list) and len(findings) == 2, findings
    assert all("[family-2]" in f and "'fellows'" in f for f in findings), findings
    assert any("bad_transcription.md" in f for f in findings), findings
    assert any("bad_wrapped.md" in f for f in findings), findings
    assert not any("good_transcription.md" in f for f in findings), findings


# --------------------------------------------------------------------------
# T13: the +/-1 physical-line GRAIN, pinned (v4 MF1)
# --------------------------------------------------------------------------


def test_t13_plus_minus_one_line_grain(tmp_path: Path) -> None:
    root = _mk_repo(tmp_path)
    # Vocabulary on N-1 (the 10-step-6.md:155-156 live shape).
    _plant(
        root,
        ".claude/rules/vocab_above.md",
        "The standing auto default order is:\nfellows-first, an rsync path, per the note\n",
    )
    # Vocabulary on N+1 (the critic-lens-reference.md:419-420 live shape).
    _plant(
        root,
        ".claude/rules/vocab_below.md",
        "legacy alias — OR is absent (the thing is\n"
        "fellows-FIRST, an rsync path), run the gate\n"
        "with the auto chain flags\n",
    )
    # Vocabulary at distance 2: OUTSIDE the window => no finding.
    _plant(
        root,
        ".claude/rules/vocab_distance2.md",
        "the auto default order:\n"
        "filler sentence with nothing\n"
        "fellows-first, an rsync path\n"
        "another filler sentence\n"
        "more filler here\n",
    )
    report = _report(root)
    findings = report["findings"]
    assert isinstance(findings, list) and len(findings) == 2, findings
    assert any("vocab_above.md:2" in f for f in findings), findings
    assert any("vocab_below.md:2" in f for f in findings), findings
    assert not any("vocab_distance2.md" in f for f in findings), findings


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
